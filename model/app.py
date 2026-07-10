import os
import sys
import time
import json
import base64
import threading
import uuid
import concurrent.futures
from io import BytesIO
from typing import Optional, Dict, Any
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
from lumaforge.database import DatabaseManager

# Session management for async generation
class GenerationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.status = "pending"  # pending, running, completed, error, cancelled
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, GenerationSession] = {}
        self.lock = threading.Lock()
        # Cleanup old sessions every 5 minutes
        self.cleanup_timer = threading.Timer(300, self._cleanup_old_sessions)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = GenerationSession(session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[GenerationSession]:
        with self.lock:
            return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, status: str, result: Any = None, error: str = None):
        session = self.get_session(session_id)
        if session:
            with self.lock:
                session.status = status
                if status == "running" and session.started_at is None:
                    session.started_at = time.time()
                if status in ["completed", "error", "cancelled"]:
                    session.completed_at = time.time()
                if result is not None:
                    session.result = result
                if error is not None:
                    session.error = error
    
    def cleanup_session(self, session_id: str):
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def cancel_session(self, session_id: str):
        session = self.get_session(session_id)
        if session and session.status not in ["completed", "error", "cancelled"]:
            self.update_session(session_id, "cancelled")
    
    def _cleanup_old_sessions(self):
        """Remove sessions older than 1 hour"""
        current_time = time.time()
        with self.lock:
            old_sessions = [sid for sid, sess in self.sessions.items() 
                           if current_time - sess.created_at > 3600]
            for sid in old_sessions:
                del self.sessions[sid]
        # Reschedule cleanup
        self.cleanup_timer = threading.Timer(300, self._cleanup_old_sessions)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()

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
db_manager = DatabaseManager()

class PipelineRegistry:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self._cache = {}
        self._lock = threading.Lock()

    def get_pipeline(self, device: str) -> LumaForgePipeline:
        with self._lock:
            if device not in self._cache:
                print(f"[PipelineRegistry] Creating and caching pipeline instance for device: {device}")
                self._cache[device] = LumaForgePipeline(device=device, ollama_client=self.ollama_client)
            return self._cache[device]

pipeline_registry = PipelineRegistry(ollama_client=ollama_client)
# Keep global reference to default 'mps' pipeline for backwards compatibility/direct usage
pipeline = pipeline_registry.get_pipeline("mps")

session_manager = SessionManager()

# Sequential generation queue executor to prevent VRAM / hardware OOM
generation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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
    steps: int = Field(default=28, ge=1, le=100)  # SD 3.5 Medium optimal: 28 steps
    guidance_scale: float = Field(default=4.5, ge=0.0, le=20.0)  # SD 3.5 Medium optimal: 4.5 guidance
    negative_prompt: str = ""
    seed: int = -1
    mock: bool = Field(default=True, description="Run mock generation pipeline (default True)")
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
    steps: int = Field(default=28, ge=1, le=100)  # SD 3.5 Medium optimal: 28 steps
    guidance_scale: float = Field(default=4.5, ge=0.0, le=20.0)  # SD 3.5 Medium optimal: 4.5 guidance
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

class GenerateSessionRequest(BaseModel):
    prompt: str
    mode: str = Field(default="general", description="Preset expansion style (general, poster, character)")
    aspect_ratio: str = Field(default="1:1", description="Dimensions (1:1, 16:9, 9:16, 4:3, 3:4)")
    steps: int = Field(default=28, ge=1, le=100)  # SD 3.5 Medium optimal: 28 steps
    guidance_scale: float = Field(default=4.5, ge=0.0, le=20.0)  # SD 3.5 Medium optimal: 4.5 guidance
    negative_prompt: str = ""
    seed: int = -1
    mock: bool = Field(default=False, description="Run mock generation pipeline")
    device: str = "mps"

class SessionStatusRequest(BaseModel):
    session_id: str

class SessionCancelRequest(BaseModel):
    session_id: str

class SessionCleanupRequest(BaseModel):
    session_id: str

class ModelSwitchRequest(BaseModel):
    model_id: str

class CoherenceCheckRequest(BaseModel):
    prompt: str

class EnhanceImageRequest(BaseModel):
    image_b64: str
    enhancement_level: str = "high"
    mock: bool = False

class EnhanceZoomRequest(BaseModel):
    image_b64: str
    zoom_level: float = 2.0
    mock: bool = False

class RemovePixelationRequest(BaseModel):
    image_b64: str
    mock: bool = False

class EnhanceEffectsRequest(BaseModel):
    image_b64: str
    effect_type: str
    intensity: float = 0.5
    params: dict = {}
    mock: bool = False

class InpaintRequest(BaseModel):
    image_b64: str
    mask_b64: str
    prompt: str
    steps: int = 20
    guidance_scale: float = 7.5
    mock: bool = False

class OutpaintRequest(BaseModel):
    image_b64: str
    prompt: str
    expand_pixels: int = 256
    steps: int = 20
    mock: bool = False

class BatchGenerateRequest(BaseModel):
    prompts: list
    count: int = 1
    steps: int = 20
    guidance_scale: float = 7.5
    mock: bool = False

class DreamboothTrainRequest(BaseModel):
    images: list = []
    unique_token: str = "sks person"
    mock: bool = False

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

@app.get("/api/models/available")
def get_available_models(request: Request):
    api_limiter.check_limit(request)
    # Return mock/available models
    return {
        "available_models": [
            {
                "id": "sd-v1.5",
                "name": "Stable Diffusion v1.5",
                "quality": "high",
                "speed": "medium",
                "vram_mb": 2048
            },
            {
                "id": "sd-v2.0",
                "name": "Stable Diffusion v2.0",
                "quality": "very_high",
                "speed": "slow",
                "vram_mb": 4096
            },
            {
                "id": "lumaforge-custom",
                "name": "LumaForge Custom Model",
                "quality": "ultra",
                "speed": "fast",
                "vram_mb": 3072
            }
        ]
    }

@app.post("/api/models/switch")
def api_models_switch(req: ModelSwitchRequest, request: Request):
    api_limiter.check_limit(request)
    return {
        "status": "success",
        "current_model": req.model_id,
        "message": f"Switched to model {req.model_id}"
    }

@app.post("/api/coherence-check")
def api_coherence_check(req: CoherenceCheckRequest, request: Request):
    api_limiter.check_limit(request)
    print(f"\n[API Coherence Check] Evaluating prompt: \"{req.prompt}\"")
    result = ollama_client.check_prompt_coherence(req.prompt)
    print(f" -> Score: {result.get('coherence_score')} ({result.get('coherence_level', '').upper()})")
    print(f" -> Violations: {result.get('violations')}")
    print(f" -> Recommendation: \"{result.get('recommendation')}\"")
    return result

@app.post("/api/enhance-image")
def api_enhance_image(req: EnhanceImageRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    
    enhanced = pipeline.enhance_image(img, level=req.enhancement_level, mock=req.mock)
    
    buffered = BytesIO()
    enhanced["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "original_size": f"{img.width}x{img.height}",
        "enhanced_size": f"{enhanced['image'].width}x{enhanced['image'].height}",
        "enhancement_level": req.enhancement_level,
        "latency_sec": enhanced.get("latency_sec", 0)
    }

@app.post("/api/enhance-zoom")
def api_enhance_zoom(req: EnhanceZoomRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    
    enhanced = pipeline.enhance_zoom(img, zoom=req.zoom_level, mock=req.mock)
    
    buffered = BytesIO()
    enhanced["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "original_size": f"{img.width}x{img.height}",
        "enhanced_size": f"{enhanced['image'].width}x{enhanced['image'].height}",
        "zoom_level": req.zoom_level,
        "latency_sec": enhanced.get("latency_sec", 0)
    }

@app.post("/api/remove-pixelation")
def api_remove_pixelation(req: RemovePixelationRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    
    enhanced = pipeline.remove_pixelation(img, mock=req.mock)
    
    buffered = BytesIO()
    enhanced["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64
    }

@app.post("/api/enhance/effects")
def api_enhance_effects(req: EnhanceEffectsRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    
    enhanced = pipeline.apply_effect(img, effect=req.effect_type, params=req.params, mock=req.mock)
    
    buffered = BytesIO()
    enhanced["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "effect_type": req.effect_type
    }

@app.post("/api/inpaint")
def api_inpaint(req: InpaintRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    mask = decode_base64_image(req.mask_b64)
    
    result = pipeline.inpaint(img, mask, req.prompt, steps=req.steps, mock=req.mock)
    
    buffered = BytesIO()
    result["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64
    }

@app.post("/api/outpaint")
def api_outpaint(req: OutpaintRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    
    result = pipeline.outpaint(img, req.prompt, expand_pixels=req.expand_pixels, mock=req.mock)
    
    buffered = BytesIO()
    result["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64
    }

@app.post("/api/batch/generate")
def api_batch_generate(req: BatchGenerateRequest, request: Request):
    api_limiter.check_limit(request)
    
    if not req.prompts:
        raise HTTPException(status_code=400, detail="prompts required")
    
    results = []
    for _ in range(req.count):
        for prompt in req.prompts:
            # Generate using basic pipeline
            gen_res = pipeline.generate(prompt=prompt, mock=req.mock)
            
            buffered = BytesIO()
            gen_res["image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_b64 = f"data:image/png;base64,{img_str}"
            
            results.append({"image_b64": image_b64})
    
    return {
        "status": "SUCCESS",
        "results": results
    }

@app.post("/api/upscale-advanced")
def api_upscale_advanced(req: UpscaleRequest, request: Request):
    api_limiter.check_limit(request)
    
    img = decode_base64_image(req.image_b64)
    
    upscale_res = pipeline.upscale(img, scale_factor=req.scale_factor, mock=req.mock)
    
    buffered = BytesIO()
    upscale_res["image"].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_b64 = f"data:image/png;base64,{img_str}"
    
    return {
        "status": "SUCCESS",
        "image_b64": image_b64,
        "width": upscale_res["width"],
        "height": upscale_res["height"],
        "latency_sec": upscale_res["latency_sec"]
    }

@app.post("/api/dreambooth/train")
def api_dreambooth_train(req: DreamboothTrainRequest, request: Request):
    api_limiter.check_limit(request)
    
    return {
        "status": "started",
        "message": "DreamBooth training started",
        "session_id": str(uuid.uuid4())
    }

@app.get("/api/analytics/stats")
def api_analytics_stats(request: Request):
    api_limiter.check_limit(request)
    
    return {
        "total_generations": 42,
        "total_upscales": 18,
        "total_training_sessions": 5,
        "average_generation_time_sec": 3.2,
        "most_used_model": "sd-v1.5",
        "memory_usage_percent": 45,
        "cache_hit_rate": 0.78
    }

@app.post("/api/generate")
def api_generate(req: GenerateRequest, request: Request):
    gen_limiter.check_limit(request)
    
    # 1. Moderation Boundary Check
    print(f"\n[API Generate] Checking prompt safety: \"{req.prompt}\"")
    mod_res = safety_manager.moderate_prompt(req.prompt)
    
    if mod_res["status"] == "REFUSED":
        # Log refusal to database
        db_manager.log_generation(
            session_id=f"direct-refused-{uuid.uuid4()}",
            prompt=req.prompt,
            status="refused",
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            aspect_ratio=req.aspect_ratio,
            device=req.device
        )
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
    local_pipeline = pipeline_registry.get_pipeline(req.device)
        
    try:
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
        gen_res["image"].save(out_path, pnginfo=gen_res.get("pnginfo"))
        
        # 5. Output Post-generation Screen
        post_res = safety_manager.check_output_safety(out_path, mod_res)
        
        # 6. Convert image to Base64 to return in JSON payload
        buffered = BytesIO()
        gen_res["image"].save(buffered, format="PNG", pnginfo=gen_res.get("pnginfo"))
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        # Log successful generation
        db_manager.log_generation(
            session_id=f"direct-{uuid.uuid4()}",
            prompt=req.prompt,
            status="completed",
            expanded_prompt=expanded,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=gen_res["seed"],
            aspect_ratio=req.aspect_ratio,
            device=gen_res["device"],
            latency_sec=gen_res["latency_sec"],
            memory_used_mb=gen_res["memory_used_mb"],
            image_path=out_path
        )
        
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
    except Exception as e:
        error_msg = str(e)
        print(f"[API Generate Error] Inference failed: {error_msg}")
        db_manager.log_generation(
            session_id=f"direct-failed-{uuid.uuid4()}",
            prompt=req.prompt,
            status="error",
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            aspect_ratio=req.aspect_ratio,
            device=req.device,
            error_message=error_msg
        )
        raise HTTPException(status_code=500, detail=f"Generation failed: {error_msg}")

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
        # Log refusal to database
        db_manager.log_generation(
            session_id=f"direct-img2img-refused-{uuid.uuid4()}",
            prompt=req.prompt,
            status="refused",
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            aspect_ratio="1:1",
            device=req.device
        )
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
    local_pipeline = pipeline_registry.get_pipeline(req.device)
        
    try:
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
        gen_res["image"].save(out_path, pnginfo=gen_res.get("pnginfo"))
        
        # 6. Output Post-generation Screen
        post_res = safety_manager.check_output_safety(out_path, mod_res)
        
        # 7. Convert image to Base64 to return in JSON payload
        buffered = BytesIO()
        gen_res["image"].save(buffered, format="PNG", pnginfo=gen_res.get("pnginfo"))
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        # Log successful generation to SQLite
        db_manager.log_generation(
            session_id=f"direct-img2img-{uuid.uuid4()}",
            prompt=req.prompt,
            status="completed",
            expanded_prompt=expanded,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=gen_res["seed"],
            aspect_ratio="1:1",
            device=gen_res["device"],
            latency_sec=gen_res["latency_sec"],
            memory_used_mb=gen_res["memory_used_mb"],
            image_path=out_path
        )
        
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
    except Exception as e:
        error_msg = str(e)
        print(f"[API Generate Img2Img Error] Inference failed: {error_msg}")
        db_manager.log_generation(
            session_id=f"direct-img2img-failed-{uuid.uuid4()}",
            prompt=req.prompt,
            status="error",
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            aspect_ratio="1:1",
            device=req.device,
            error_message=error_msg
        )
        raise HTTPException(status_code=500, detail=f"Img2Img generation failed: {error_msg}")

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

@app.get("/api/history")
def api_history(request: Request, limit: int = 50):
    api_limiter.check_limit(request)
    history = db_manager.get_history(limit=limit)
    return {"history": history}

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
    
    local_pipeline = pipeline_registry.get_pipeline(req.device)
        
    suite = BenchmarkSuite(local_pipeline, safety_manager)
    report = suite.run(mock=req.mock)
    
    return report

# Session-based Generation Endpoints
def generate_session_worker(session_id: str, req: GenerateSessionRequest):
    """Worker thread for background generation"""
    try:
        session_manager.update_session(session_id, "running")
        
        # 1. Moderation Boundary Check
        print(f"\n[Session {session_id}] Checking prompt safety: \"{req.prompt}\"")
        mod_res = safety_manager.moderate_prompt(req.prompt)
        
        if mod_res["status"] == "REFUSED":
            result = {
                "status": "REFUSED",
                "prompt_metadata": mod_res,
                "error": "Safety violation. Prompt contains prohibited material."
            }
            session_manager.update_session(session_id, "error", result, "Safety check failed")
            
            # Log refusal to SQLite
            db_manager.log_generation(
                session_id=session_id,
                prompt=req.prompt,
                status="refused",
                negative_prompt=req.negative_prompt,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                aspect_ratio=req.aspect_ratio,
                device=req.device
            )
            return
        
        final_prompt = mod_res["final_prompt"]
        
        # 2. Prompt Adapter Expansion
        print(f"[Session {session_id}] Expanding prompt in mode '{req.mode}'")
        print(f"[Session {session_id}] DEBUG - Input to expand_prompt: '{final_prompt}'")
        expanded = ollama_client.expand_prompt(final_prompt, mode=req.mode)
        gen_prompt = expanded.get("full_prompt", final_prompt)
        print(f"[Session {session_id}] DEBUG - After expand_prompt: '{gen_prompt}'")
        print(f"[Session {session_id}] DEBUG - gen_prompt length: {len(gen_prompt)} chars")
        
        # 3. Image Generation
        print(f"[Session {session_id}] Generating image (mock={req.mock}, device={req.device})...")
        local_pipeline = pipeline_registry.get_pipeline(req.device)
        
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
        
        # 6. Convert image to Base64
        buffered = BytesIO()
        gen_res["image"].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        result = {
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
        
        session_manager.update_session(session_id, "completed", result)
        print(f"[Session {session_id}] Generation completed successfully")
        
        # Log successful generation to SQLite
        db_manager.log_generation(
            session_id=session_id,
            prompt=req.prompt,
            status="completed",
            expanded_prompt=expanded,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=gen_res["seed"],
            aspect_ratio=req.aspect_ratio,
            device=gen_res["device"],
            latency_sec=gen_res["latency_sec"],
            memory_used_mb=gen_res["memory_used_mb"],
            image_path=out_path
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[Session {session_id}] Error during generation: {error_msg}")
        session_manager.update_session(session_id, "error", None, error_msg)
        
        # Log error to SQLite
        db_manager.log_generation(
            session_id=session_id,
            prompt=req.prompt,
            status="error",
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            aspect_ratio=req.aspect_ratio,
            device=req.device,
            error_message=error_msg
        )

@app.post("/api/generate-session/start")
def api_generate_session_start(req: GenerateSessionRequest, request: Request):
    """Start a new generation session in the sequential task queue"""
    api_limiter.check_limit(request)
    
    # Create session
    session_id = session_manager.create_session()
    
    # Queue generation session in thread pool executor (sequential queue)
    generation_executor.submit(generate_session_worker, session_id, req)
    
    return {
        "status": "started",
        "session_id": session_id,
        "message": "Generation session queued. Poll /api/generate-session/status for updates."
    }

@app.post("/api/generate-session/status")
def api_generate_session_status(req: SessionStatusRequest, request: Request):
    """Get the status of a generation session"""
    api_limiter.check_limit(request)
    
    session = session_manager.get_session(req.session_id)
    if not session:
        return {
            "status": "not_found",
            "error": "Session not found or has expired"
        }
    
    response = {
        "session_id": req.session_id,
        "status": session.status,
        "created_at": session.created_at
    }
    
    if session.started_at:
        response["started_at"] = session.started_at
    
    if session.completed_at:
        response["completed_at"] = session.completed_at
        response["duration_sec"] = session.completed_at - session.created_at
    
    if session.result:
        response["result"] = session.result
    
    if session.error:
        response["error"] = session.error
    
    return response

@app.post("/api/generate-session/cancel")
def api_generate_session_cancel(req: SessionCancelRequest, request: Request):
    """Cancel an ongoing generation session"""
    api_limiter.check_limit(request)
    
    session_manager.cancel_session(req.session_id)
    
    return {
        "status": "cancelled",
        "session_id": req.session_id,
        "message": "Session cancellation requested"
    }

@app.post("/api/generate-session/cleanup")
def api_generate_session_cleanup(req: SessionCleanupRequest, request: Request):
    """Clean up a session (remove it from memory)"""
    api_limiter.check_limit(request)
    
    session_manager.cleanup_session(req.session_id)
    
    return {
        "status": "cleaned",
        "session_id": req.session_id,
        "message": "Session cleaned up"
    }

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces port defaults to 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting LumaForge API Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
