import os
import sys
import time
import json
import base64
import threading
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure model directory is in Python path for absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lumaforge.ollama_client import OllamaClient
from lumaforge.pipeline import LumaForgePipeline
from lumaforge.safety import SafetyManager
from lumaforge.benchmark import BenchmarkSuite
from lumaforge.dataset_curator import DatasetCurator
from lumaforge.train import LumaForgeTrainer

app = FastAPI(
    title="LumaForge AuraGen MPS API",
    description="Backend API engine for image generation, fine-tuning, and audit logs.",
    version="1.0.0"
)

# Enable CORS for the separate Next.js web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to web client domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons for backend resources
ollama_client = OllamaClient()
safety_manager = SafetyManager(ollama_client=ollama_client)
pipeline = LumaForgePipeline(device="mps")

# Background training tracking
training_thread = None

# Custom in-memory rate limiter to avoid redis dependencies on Hugging Face Spaces
class RateLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests = {} # ip -> list of timestamps
        self.lock = threading.Lock()

    def check_limit(self, request: Request):
        ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        with self.lock:
            if ip not in self.requests:
                self.requests[ip] = []
            
            # Filter timestamps outside the sliding window
            self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
            
            if len(self.requests[ip]) >= self.limit:
                retry_after = int(self.window - (now - self.requests[ip][0]))
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Too Many Requests",
                        "message": f"Rate limit exceeded. Please wait {retry_after} seconds.",
                        "retry_after": retry_after
                    }
                )
            self.requests[ip].append(now)

# Limiters: 10 generations per minute, 60 requests per minute for other api endpoints
gen_limiter = RateLimiter(limit=10, window=60)
api_limiter = RateLimiter(limit=60, window=60)

# Request Models
class GenerateRequest(BaseModel):
    prompt: str
    mode: str = Field(default="general", description="Preset expansion style (general, poster, character)")
    aspect_ratio: str = Field(default="1:1", description="Dimensions (1:1, 16:9, 9:16, 4:3, 3:4)")
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    negative_prompt: str = ""
    seed: int = -1
    mock: bool = Field(default=False, description="Run mock generation pipeline")
    device: str = "mps"

class TrainRequest(BaseModel):
    epochs: int = 3
    lr: float = 5e-6
    batch_size: int = 2
    demo: bool = True
    cooldown: float = 0.0
    checkpoint_steps: int = 0
    resume: bool = False
    checkpoint_dir: str = "weights/checkpoints"

class CurateRequest(BaseModel):
    limit: int = 90
    caption: bool = True

class BenchmarkRequest(BaseModel):
    mock: bool = True
    device: str = "mps"

class Img2ImgRequest(BaseModel):
    prompt: str
    image_b64: str
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: str = Field(default="general", description="Preset expansion style (general, poster, character)")
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    negative_prompt: str = ""
    seed: int = -1
    mock: bool = Field(default=False, description="Run mock generation pipeline")
    device: str = "mps"

class UpscaleRequest(BaseModel):
    image_b64: str
    scale_factor: float = Field(default=2.0, ge=1.0, le=4.0)
    mock: bool = Field(default=False)

class RemoveBackgroundRequest(BaseModel):
    image_b64: str
    mock: bool = Field(default=False)

class ColorizeRequest(BaseModel):
    image_b64: str
    style: str = Field(default="vibrant", description="Colorization style: vibrant, warm, cool, vintage, sepia")
    mock: bool = Field(default=False)

class FaceRestorationRequest(BaseModel):
    image_b64: str
    intensity: str = Field(default="medium", description="Restoration intensity: low, medium, high, ultra")
    mock: bool = Field(default=False)

# Endpoints
@app.get("/api/status")
def get_status(request: Request):
    api_limiter.check_limit(request)
    import torch
    
    ollama_ok = ollama_client.check_connection()
    mps_ok = torch.backends.mps.is_available()
    device = "mps" if mps_ok else "cpu"
    
    return {
        "status": "healthy",
        "device": device,
        "mps_available": mps_ok,
        "ollama_connected": ollama_ok,
        "backend": "FastAPI + PyTorch",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.post("/api/generate")
def api_generate(req: GenerateRequest, request: Request):
    gen_limiter.check_limit(request)
    
    # 1. Moderation Boundary Check
    print(f"\n[API Generate] Checking prompt safety: \"{req.prompt}\"")
    mod_res = safety_manager.moderate_prompt(req.prompt)
    
    if mod_res["status"] == "REFUSED":
        return {
            "status": "REFUSED",
            "prompt_metadata": mod_res,
            "error": "Safety violation. Prompt contains prohibited material."
        }
        
    final_prompt = mod_res["final_prompt"]
    
    # 2. Prompt Adapter Expansion
    print(f"[API Generate] Expanding prompt in mode '{req.mode}'")
    expanded = ollama_client.expand_prompt(final_prompt, mode=req.mode)
    gen_prompt = expanded.get("full_prompt", final_prompt)
    
    # 3. Image Generation
    print(f"[API Generate] Generating image (mock={req.mock}, device={req.device})...")
    # If device matches our pipeline device, use existing pipeline, otherwise initialize
    local_pipeline = pipeline
    if req.device != pipeline.device:
        local_pipeline = LumaForgePipeline(device=req.device)
        
    gen_res = local_pipeline.generate(
        prompt=gen_prompt,
        aspect_ratio=req.aspect_ratio,
        steps=req.steps,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        negative_prompt=req.negative_prompt,
        mock=req.mock
    )
    
    # 4. Save locally for record-keeping and post-safety checks
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"output_{gen_res['seed']}.png")
    gen_res["image"].save(out_path)
    
    # 5. Output Post-generation Screen
    post_res = safety_manager.check_output_safety(out_path, mod_res)
    
    # 6. Convert image to Base64 to return in JSON payload
    buffered = BytesIO()
    gen_res["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": mod_res["status"],
        "image_b64": image_b64,
        "prompt_metadata": mod_res,
        "expanded_prompt": expanded,
        "generation_metadata": {
            "latency_sec": gen_res["latency_sec"],
            "memory_used_mb": gen_res["memory_used_mb"],
            "seed": gen_res["seed"],
            "width": gen_res["width"],
            "height": gen_res["height"],
            "device": gen_res["device"],
            "used_mock": gen_res["used_mock"]
        },
        "safety_check": post_res
    }

def decode_base64_image(image_b64: str) -> Image.Image:
    try:
        from PIL import Image
        if "," in image_b64:
            header, image_b64 = image_b64.split(",", 1)
        data = base64.b64decode(image_b64)
        return Image.open(BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

@app.post("/api/generate-img2img")
def api_generate_img2img(req: Img2ImgRequest, request: Request):
    gen_limiter.check_limit(request)
    
    # 1. Moderation Boundary Check
    print(f"\n[API Generate Img2Img] Checking prompt safety: \"{req.prompt}\"")
    mod_res = safety_manager.moderate_prompt(req.prompt)
    
    if mod_res["status"] == "REFUSED":
        return {
            "status": "REFUSED",
            "prompt_metadata": mod_res,
            "error": "Safety violation. Prompt contains prohibited material."
        }
        
    final_prompt = mod_res["final_prompt"]
    
    # 2. Prompt Adapter Expansion
    print(f"[API Generate Img2Img] Expanding prompt in mode '{req.mode}'")
    expanded = ollama_client.expand_prompt(final_prompt, mode=req.mode)
    gen_prompt = expanded.get("full_prompt", final_prompt)
    
    # 3. Decode base64 input image
    img = decode_base64_image(req.image_b64)
    
    # 4. Image Generation
    print(f"[API Generate Img2Img] Generating image (mock={req.mock}, device={req.device}, strength={req.strength})...")
    local_pipeline = pipeline
    if req.device != pipeline.device:
        local_pipeline = LumaForgePipeline(device=req.device)
        
    gen_res = local_pipeline.generate_img2img(
        image=img,
        prompt=gen_prompt,
        strength=req.strength,
        steps=req.steps,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        negative_prompt=req.negative_prompt,
        mock=req.mock
    )
    
    # 5. Save locally for record-keeping and post-safety checks
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"output_{gen_res['seed']}.png")
    gen_res["image"].save(out_path)
    
    # 6. Output Post-generation Screen
    post_res = safety_manager.check_output_safety(out_path, mod_res)
    
    # 7. Convert image to Base64 to return in JSON payload
    buffered = BytesIO()
    gen_res["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": mod_res["status"],
        "image_b64": image_b64,
        "prompt_metadata": mod_res,
        "expanded_prompt": expanded,
        "generation_metadata": {
            "latency_sec": gen_res["latency_sec"],
            "memory_used_mb": gen_res["memory_used_mb"],
            "seed": gen_res["seed"],
            "width": gen_res["width"],
            "height": gen_res["height"],
            "steps": gen_res["steps"],
            "guidance_scale": gen_res["guidance_scale"],
            "strength": gen_res["strength"],
            "device": gen_res["device"],
            "used_mock": gen_res["used_mock"]
        },
        "safety_check": post_res
    }

@app.post("/api/upscale")
def api_upscale(req: UpscaleRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Upscale] Upscaling image (mock={req.mock}, scale_factor={req.scale_factor})...")
    img = decode_base64_image(req.image_b64)
    
    upscale_res = pipeline.upscale(img, scale_factor=req.scale_factor, mock=req.mock)
    
    # Convert back to Base64
    buffered = BytesIO()
    upscale_res["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "width": upscale_res["width"],
        "height": upscale_res["height"],
        "latency_sec": upscale_res["latency_sec"],
        "memory_used_mb": upscale_res["memory_used_mb"],
    }

@app.post("/api/remove-background")
def api_remove_background(req: RemoveBackgroundRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Remove Background] Removing background (mock={req.mock})...")
    img = decode_base64_image(req.image_b64)
    
    out_img = pipeline.remove_background(img, mock=req.mock)
    
    # Convert to Base64 (PNG to support transparency!)
    buffered = BytesIO()
    out_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64
    }

@app.post("/api/colorize")
def api_colorize(req: ColorizeRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Colorize] Colorizing image (style={req.style}, mock={req.mock})...")
    img = decode_base64_image(req.image_b64)
    
    colorized = pipeline.colorize(img, style=req.style, mock=req.mock)
    
    # Convert to Base64
    buffered = BytesIO()
    colorized["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "style": req.style,
        "latency_sec": colorized.get("latency_sec", 0),
        "memory_used_mb": colorized.get("memory_used_mb", 0)
    }

@app.post("/api/face-restoration")
def api_face_restoration(req: FaceRestorationRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Face Restoration] Restoring faces (intensity={req.intensity}, mock={req.mock})...")
    img = decode_base64_image(req.image_b64)
    
    restored = pipeline.restore_face(img, intensity=req.intensity, mock=req.mock)
    
    # Convert to Base64
    buffered = BytesIO()
    restored["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "intensity": req.intensity,
        "latency_sec": restored.get("latency_sec", 0),
        "memory_used_mb": restored.get("memory_used_mb", 0)
    }

@app.get("/api/audit-log")
def api_audit_log(request: Request, limit: int = 20):
    api_limiter.check_limit(request)
    logs = safety_manager.get_audit_logs(limit=limit)
    return {"logs": logs}

def run_train_worker(req: TrainRequest):
    trainer = LumaForgeTrainer(device="mps" if req.demo else "cpu")
    trainer.run_training(
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
        demo=req.demo,
        cooldown_secs=req.cooldown,
        checkpoint_steps=req.checkpoint_steps,
        resume=req.resume,
        checkpoint_dir=req.checkpoint_dir
    )

@app.post("/api/train")
def api_train(req: TrainRequest, request: Request):
    api_limiter.check_limit(request)
    global training_thread
    
    if training_thread and training_thread.is_alive():
        raise HTTPException(
            status_code=400,
            detail="Model fine-tuning is currently running in the background."
        )
        
    training_thread = threading.Thread(target=run_train_worker, args=(req,))
    training_thread.start()
    
    return {
        "status": "started",
        "message": "Fine-tuning job successfully launched in background.",
        "params": req.dict()
    }

@app.get("/api/train/status")
def api_train_status(request: Request):
    api_limiter.check_limit(request)
    log_path = "train_log.json"
    
    is_active = training_thread is not None and training_thread.is_alive()
    
    if not os.path.exists(log_path):
        return {
            "status": "IDLE" if not is_active else "RUNNING",
            "epoch": 0,
            "total_epochs": 0,
            "progress_pct": 0.0,
            "metrics": {"train_loss": 0.0, "val_loss": 0.0, "prompt_adherence": 0.0},
            "history": []
        }
        
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
        # Ensure correct run state status
        if is_active:
            data["status"] = "RUNNING"
        else:
            if data.get("status") == "RUNNING":
                data["status"] = "COMPLETED"
        return data
    except Exception as e:
        return {"error": f"Failed to read train log: {str(e)}", "status": "RUNNING" if is_active else "IDLE"}

@app.post("/api/curate")
def api_curate(req: CurateRequest, request: Request):
    api_limiter.check_limit(request)
    curator = DatasetCurator()
    count = curator.download_and_curate(limit=req.limit, use_ollama_captioning=req.caption)
    return {"curated_count": count}

@app.post("/api/benchmark")
def api_benchmark(req: BenchmarkRequest, request: Request):
    api_limiter.check_limit(request)
    
    # Run in a simple separate execution or directly
    local_pipeline = pipeline
    if req.device != pipeline.device:
        local_pipeline = LumaForgePipeline(device=req.device)
        
    suite = BenchmarkSuite(local_pipeline, safety_manager)
    report = suite.run(mock=req.mock)
    
    return report

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces port defaults to 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting LumaForge API Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
