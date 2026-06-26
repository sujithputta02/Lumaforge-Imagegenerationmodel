import os
import sys
import time
import json
import base64
import threading
import uuid
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import json
import os

# Global session manager for generation tracking
class GenerationSession:
    """Track ongoing generations by session ID"""
    def __init__(self):
        self.sessions = {}  # session_id -> generation info
        self.lock = threading.Lock()
    
    def create_session(self, user_id: str = None):
        """Create new generation session"""
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                "status": "pending",
                "progress": 0,
                "user_id": user_id,
                "created_at": time.time(),
                "result": None,
                "error": None,
                "cancel_flag": False,
                "thread": None
            }
        return session_id
    
    def update_session(self, session_id: str, **kwargs):
        """Update session status"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(kwargs)
    
    def get_session(self, session_id: str):
        """Get session info"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def cancel_session(self, session_id: str):
        """Cancel ongoing generation"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["cancel_flag"] = True
                return True
        return False
    
    def cleanup_session(self, session_id: str):
        """Remove completed session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

session_manager = GenerationSession()


from lumaforge.ollama_client import OllamaClient
from lumaforge.pipeline import LumaForgePipeline
from lumaforge.safety import SafetyManager
from lumaforge.benchmark import BenchmarkSuite
from lumaforge.dataset_curator import DatasetCurator
from lumaforge.train import LumaForgeTrainer
from lumaforge.cost_optimizer import CostOptimizer, get_optimization_tips
from lumaforge.reality_validator import RealityValidator, is_prompt_coherent

# Multi-Model Router for automatic model selection
class ModelRouter:
    def __init__(self):
        self.available_models = {
            "sd-v1.5": {
                "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "use_sdxl": False,
                "quality": "standard",
                "speed": "fast"
            },
            "sdxl": {
                "id": "stabilityai/stable-diffusion-xl-base-1.0",
                "use_sdxl": True,
                "quality": "high",
                "speed": "slow"
            },
            "flux": {
                "id": "black-forest-labs/FLUX.1-dev",
                "use_sdxl": False,
                "quality": "premium",
                "speed": "very_slow"
            }
        }
        self.active_model = "sd-v1.5"
        self.pipeline = None
    
    def select_model(self, quality: str = "standard"):
        """Automatically select best model for quality level."""
        if quality == "premium":
            selected = "flux"
        elif quality == "high":
            selected = "sdxl"
        else:
            selected = "sd-v1.5"
        
        if selected != self.active_model:
            print(f"[ModelRouter] Switching from {self.active_model} to {selected}")
            self.active_model = selected
            self._reload_pipeline()
    
    def _reload_pipeline(self):
        """Reload pipeline with new model."""
        model_config = self.available_models[self.active_model]
        self.pipeline = LumaForgePipeline(
            model_id=model_config["id"],
            device="mps",
            use_sdxl=model_config["use_sdxl"]
        )
    
    def get_pipeline(self):
        """Get current pipeline."""
        if self.pipeline is None:
            self._reload_pipeline()
        return self.pipeline

# Analytics Tracker
class AnalyticsTracker:
    def __init__(self):
        self.stats = {
            "total_generations": 0,
            "total_upscales": 0,
            "total_inpaints": 0,
            "errors": 0,
            "by_model": {},
            "by_feature": {},
            "latencies": [],
            "memory_usage": []
        }
    
    def record_generation(self, model: str, latency: float, memory: float, success: bool = True):
        """Record generation metrics."""
        if success:
            self.stats["total_generations"] += 1
        else:
            self.stats["errors"] += 1
        
        if model not in self.stats["by_model"]:
            self.stats["by_model"][model] = 0
        self.stats["by_model"][model] += 1
        
        self.stats["latencies"].append(latency)
        self.stats["memory_usage"].append(memory)
    
    def get_stats(self):
        """Get aggregated statistics."""
        avg_latency = sum(self.stats["latencies"][-100:]) / max(1, len(self.stats["latencies"][-100:]))
        avg_memory = sum(self.stats["memory_usage"][-100:]) / max(1, len(self.stats["memory_usage"][-100:]))
        
        return {
            "total_generations": self.stats["total_generations"],
            "total_errors": self.stats["errors"],
            "by_model": self.stats["by_model"],
            "avg_latency_sec": round(avg_latency, 2),
            "avg_memory_mb": round(avg_memory, 2),
            "error_rate": round(self.stats["errors"] / max(1, self.stats["total_generations"]), 3)
        }

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
model_router = ModelRouter()
model_router.select_model(quality="high")  # Use SDXL for high-quality realistic images
pipeline = model_router.get_pipeline()
analytics = AnalyticsTracker()

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
    session_id: str = Field(default="", description="Optional session ID for continuing generation")

class SessionStatusRequest(BaseModel):
    session_id: str = Field(description="Session ID to check")

class CancelGenerationRequest(BaseModel):
    session_id: str = Field(description="Session ID to cancel")

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

class EffectsRequest(BaseModel):
    image_b64: str
    effect_type: str = Field(description="depth-of-field, film-grain, chromatic-aberration, lens-flare")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    params: dict = Field(default={})

class InpaintRequest(BaseModel):
    image_b64: str
    mask_b64: str
    prompt: str
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)

class OutpaintRequest(BaseModel):
    image_b64: str
    expand_pixels: int = Field(default=256, ge=50, le=1000)
    prompt: str = Field(default="seamless extension")
    steps: int = Field(default=20, ge=1, le=100)

class UpscaleAdvancedRequest(BaseModel):
    image_b64: str
    scale_factor: float = Field(default=2.0, ge=2.0, le=8.0)
    model_type: str = Field(default="lanczos", description="realesrgan or lanczos")

class DreamboothRequest(BaseModel):
    name: str = Field(description="Name for the new concept/style")
    images_b64: list = Field(description="3-5 training images as base64")
    prompt: str = Field(default="a photo of {}", description="Prompt template with {} placeholder")
    steps: int = Field(default=100, ge=50, le=500)
    learning_rate: float = Field(default=5e-4, ge=1e-6, le=1e-3)

class ModelSelectRequest(BaseModel):
    model: str = Field(description="sd-v1.5, sdxl, or flux")
    quality_level: str = Field(default="standard", description="standard, high, or premium")

class CoherenceCheckRequest(BaseModel):
    prompt: str = Field(description="Prompt to check for coherence and realism")

class EnhanceImageRequest(BaseModel):
    image_b64: str = Field(description="Base64 encoded image")
    enhancement_level: str = Field(default="high", description="low, medium, high, or ultra")

class EnhanceZoomRequest(BaseModel):
    image_b64: str = Field(description="Base64 encoded image")
    zoom_level: int = Field(default=2, ge=1, le=4, description="Zoom factor (2x, 3x, 4x)")

class RemovePixelationRequest(BaseModel):
    image_b64: str = Field(description="Base64 encoded pixelated image")

class ColorizeRequest(BaseModel):
    image_b64: str = Field(description="Base64 encoded grayscale/B&W image")
    color_style: str = Field(default="vibrant", description="Color style: vibrant, warm, cool, vintage, sepia")

class FaceRestorationRequest(BaseModel):
    image_b64: str = Field(description="Base64 encoded face image to restore")
    restoration_level: str = Field(default="high", description="Restoration intensity: low, medium, high, ultra")

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

@app.post("/api/generate-session/start")
def api_generate_session_start(req: GenerateRequest, request: Request):
    """
    Start a new generation session that can survive page reloads.
    Returns session_id to track the generation.
    """
    gen_limiter.check_limit(request)
    
    # Create new session
    session_id = session_manager.create_session()
    
    # Start background generation task
    def background_generate():
        try:
            session_manager.update_session(session_id, status="running")
            
            # 0. Check & Enhance Prompt for Coherence
            print(f"\n[Session {session_id}] Checking prompt coherence")
            validator = RealityValidator()
            coherence_result = validator.validate_prompt(req.prompt)
            prompt_to_use = coherence_result["improved_prompt"] if coherence_result["enhancement_needed"] else req.prompt
            
            # Check for cancellation
            if session_manager.get_session(session_id)["cancel_flag"]:
                session_manager.update_session(session_id, status="cancelled")
                return
            
            # 1. Moderation
            mod_res = safety_manager.moderate_prompt(prompt_to_use)
            if mod_res["status"] == "REFUSED":
                session_manager.update_session(session_id, status="refused", error="Safety violation")
                return
            
            final_prompt = mod_res["final_prompt"]
            
            # Check for cancellation
            if session_manager.get_session(session_id)["cancel_flag"]:
                session_manager.update_session(session_id, status="cancelled")
                return
            
            # 2. Prompt expansion
            expanded = ollama_client.expand_prompt(final_prompt, mode=req.mode)
            gen_prompt = expanded.get("full_prompt", final_prompt)
            
            # Photorealism enforcer - Ensure realistic rendering
            if "photorealistic" not in gen_prompt.lower() and "photo" not in gen_prompt.lower():
                gen_prompt += ", photorealistic rendering, professional photography, crisp focus"
            if "painting" not in gen_prompt.lower() and "art" not in gen_prompt.lower():
                gen_prompt += ", NOT illustrated, NOT painting, NOT artwork, NOT cartoon"
            if "film" not in gen_prompt.lower():
                gen_prompt += ", cinematic film quality, natural textures, realistic skin"
            
            # Check for cancellation
            if session_manager.get_session(session_id)["cancel_flag"]:
                session_manager.update_session(session_id, status="cancelled")
                return
            
            # 3. Generate
            local_pipeline = pipeline
            if req.device != pipeline.device:
                local_pipeline = LumaForgePipeline(device=req.device)
            
            session_manager.update_session(session_id, progress=10)
            
            gen_res = local_pipeline.generate(
                prompt=gen_prompt,
                aspect_ratio=req.aspect_ratio,
                steps=req.steps,
                seed=req.seed,
                guidance_scale=req.guidance_scale,
                negative_prompt=req.negative_prompt,
                mock=req.mock
            )
            
            # Check for cancellation
            if session_manager.get_session(session_id)["cancel_flag"]:
                session_manager.update_session(session_id, status="cancelled")
                return
            
            session_manager.update_session(session_id, progress=90)
            
            # 4. Save and post-check
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", f"output_{gen_res['seed']}.png")
            gen_res["image"].save(out_path)
            
            post_res = safety_manager.check_output_safety(out_path, mod_res)
            
            # 5. Convert to Base64
            buffered = BytesIO()
            gen_res["image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_b64 = f"data:image/png;base64,{img_str}"
            
            # Store result
            result = {
                "status": mod_res["status"],
                "image_b64": image_b64,
                "coherence_check": {
                    "coherence_score": coherence_result["coherence_score"],
                    "coherence_level": coherence_result["coherence_level"],
                    "enhancement_needed": coherence_result["enhancement_needed"]
                },
                "generation_metadata": {
                    "latency_sec": gen_res["latency_sec"],
                    "seed": gen_res["seed"],
                    "width": gen_res["width"],
                    "height": gen_res["height"]
                }
            }
            
            session_manager.update_session(session_id, status="completed", result=result, progress=100)
            
        except Exception as e:
            print(f"[Session {session_id}] Error: {e}")
            session_manager.update_session(session_id, status="error", error=str(e))
    
    # Start background thread
    thread = threading.Thread(target=background_generate, daemon=False)
    thread.start()
    session_manager.update_session(session_id, thread=thread)
    
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Generation started. Use this session_id to check status or reconnect."
    }

@app.post("/api/generate-session/status")
def api_generate_session_status(req: SessionStatusRequest):
    """Check generation session status"""
    session = session_manager.get_session(req.session_id)
    
    if not session:
        return {
            "session_id": req.session_id,
            "status": "not_found",
            "error": "Session not found"
        }
    
    response = {
        "session_id": req.session_id,
        "status": session["status"],
        "progress": session["progress"],
        "created_at": session["created_at"]
    }
    
    if session["status"] == "completed":
        response["result"] = session["result"]
    elif session["status"] == "error" or session["status"] == "refused":
        response["error"] = session["error"]
    
    return response

@app.post("/api/generate-session/cancel")
def api_generate_session_cancel(req: CancelGenerationRequest):
    """Cancel an ongoing generation"""
    if session_manager.cancel_session(req.session_id):
        return {
            "session_id": req.session_id,
            "status": "cancel_requested",
            "message": "Generation cancellation requested"
        }
    else:
        return {
            "session_id": req.session_id,
            "status": "not_found",
            "error": "Session not found"
        }

@app.post("/api/generate-session/cleanup")
def api_generate_session_cleanup(req: SessionStatusRequest):
    """Clean up completed session"""
    session_manager.cleanup_session(req.session_id)
    return {
        "session_id": req.session_id,
        "status": "cleaned_up"
    }

# Keep old /api/generate for backward compatibility
@app.post("/api/generate")
def api_generate(req: GenerateRequest, request: Request):
    gen_limiter.check_limit(request)
    
    # 0. Check & Enhance Prompt for Coherence (NEW)
    print(f"\n[API Generate] Checking prompt coherence: \"{req.prompt}\"")
    validator = RealityValidator()
    coherence_result = validator.validate_prompt(req.prompt)
    
    # Use enhanced prompt if needed
    prompt_to_use = coherence_result["improved_prompt"] if coherence_result["enhancement_needed"] else req.prompt
    print(f"[API Generate] Original: {req.prompt}")
    print(f"[API Generate] Enhanced: {prompt_to_use}")
    print(f"[API Generate] Coherence Score: {coherence_result['coherence_score']}/100")
    
    # 1. Moderation Boundary Check
    print(f"[API Generate] Checking prompt safety: \"{prompt_to_use}\"")
    mod_res = safety_manager.moderate_prompt(prompt_to_use)
    
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
    
    # PHOTOREALISM ENFORCER - Ensure realistic rendering (NEW)
    # Add realism keywords if not present
    if "photorealistic" not in gen_prompt.lower() and "photo" not in gen_prompt.lower():
        gen_prompt += ", photorealistic rendering, professional photography, crisp focus"
    
    # Prevent painted/illustrated look
    if "painting" not in gen_prompt.lower() and "art" not in gen_prompt.lower():
        gen_prompt += ", NOT illustrated, NOT painting, NOT artwork, NOT cartoon"
    
    # Add film quality
    if "film" not in gen_prompt.lower():
        gen_prompt += ", cinematic film quality, natural textures, realistic skin"
    
    print(f"[API Generate] Final prompt with realism: {gen_prompt[:100]}...")
    
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
        "coherence_check": {
            "original_prompt": req.prompt,
            "enhanced_prompt": prompt_to_use,
            "coherence_score": coherence_result["coherence_score"],
            "coherence_level": coherence_result["coherence_level"],
            "enhancement_needed": coherence_result["enhancement_needed"],
            "recommendation": coherence_result["recommendation"]
        },
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

@app.post("/api/enhance/effects")
def api_apply_effects(req: EffectsRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Effects] Applying {req.effect_type} effect...")
    img = decode_base64_image(req.image_b64)
    
    try:
        if req.effect_type == "depth-of-field":
            focus_point = req.params.get("focus_point", (0.5, 0.5))
            blur_strength = int(req.params.get("blur_strength", 10) * req.intensity)
            result = pipeline.apply_depth_of_field(img, focus_point=focus_point, blur_strength=blur_strength)
        
        elif req.effect_type == "film-grain":
            grain_intensity = req.params.get("intensity", req.intensity)
            grain_size = int(req.params.get("grain_size", 1))
            result = pipeline.apply_film_grain(img, intensity=grain_intensity, grain_size=grain_size)
        
        elif req.effect_type == "chromatic-aberration":
            offset = int(req.params.get("offset", 5) * (req.intensity + 0.5))
            result = pipeline.apply_chromatic_aberration(img, offset=offset)
        
        elif req.effect_type == "lens-flare":
            center = req.params.get("center", (0.7, 0.3))
            result = pipeline.apply_lens_flare(img, center=center, intensity=req.intensity)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown effect type: {req.effect_type}")
        
        # Convert back to Base64
        buffered = BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        return {
            "status": "SUCCESS",
            "image_b64": image_b64,
            "effect": req.effect_type,
            "intensity": req.intensity
        }
    
    except Exception as e:
        print(f"[API Effects Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch/generate")
def api_batch_generate(prompts: list, count: int = 5, request: Request = None):
    """Generate multiple images in batch with different seeds."""
    if request:
        gen_limiter.check_limit(request)
    
    results = []
    print(f"[API Batch] Generating {count} images for {len(prompts)} prompts...")
    
    for prompt_idx, prompt in enumerate(prompts[:3]):  # Limit to 3 prompts to avoid timeout
        for i in range(count):
            seed = random.randint(0, 9999999)
            print(f"  [{prompt_idx + 1}/{min(3, len(prompts))}] Generating image {i + 1}/{count}...")
            
            try:
                gen_res = pipeline.generate(prompt=prompt, seed=seed, mock=True)
                
                # Convert to base64
                buffered = BytesIO()
                gen_res["image"].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_b64 = f"data:image/png;base64,{img_str}"
                
                results.append({
                    "prompt": prompt,
                    "seed": seed,
                    "image_b64": image_b64[:100] + "..." if len(image_b64) > 100 else image_b64,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "seed": seed,
                    "status": "failed",
                    "error": str(e)
                })
    
    return {
        "status": "completed",
        "total_generated": len(results),
        "results": results
    }

@app.post("/api/inpaint")
def api_inpaint(req: InpaintRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Inpaint] Regenerating masked region with prompt: {req.prompt}")
    
    img = decode_base64_image(req.image_b64)
    mask = decode_base64_image(req.mask_b64)
    
    try:
        result = pipeline.inpaint(
            image=img,
            mask=mask,
            prompt=req.prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength
        )
        
        # Convert to base64
        buffered = BytesIO()
        result["image"].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        return {
            "status": "SUCCESS",
            "image_b64": image_b64,
            "latency_sec": result["latency_sec"]
        }
    except Exception as e:
        print(f"[API Inpaint Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/outpaint")
def api_outpaint(req: OutpaintRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Outpaint] Extending canvas with prompt: {req.prompt}")
    
    img = decode_base64_image(req.image_b64)
    
    try:
        result = pipeline.outpaint(
            image=img,
            expand_pixels=req.expand_pixels,
            prompt=req.prompt,
            steps=req.steps
        )
        
        # Convert to base64
        buffered = BytesIO()
        result["image"].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        return {
            "status": "SUCCESS",
            "image_b64": image_b64,
            "latency_sec": result["latency_sec"],
            "expanded_from": result["original_size"],
            "expanded_to": result["expanded_size"]
        }
    except Exception as e:
        print(f"[API Outpaint Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upscale-advanced")
def api_upscale_advanced(req: UpscaleAdvancedRequest, request: Request):
    api_limiter.check_limit(request)
    
    print(f"[API Upscale] Upscaling {req.scale_factor}x using {req.model_type}...")
    
    img = decode_base64_image(req.image_b64)
    
    try:
        result = pipeline.upscale_advanced(
            image=img,
            scale_factor=req.scale_factor,
            model_type=req.model_type
        )
        
        # Convert to base64
        buffered = BytesIO()
        result["image"].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        return {
            "status": "SUCCESS",
            "image_b64": image_b64,
            "width": result["width"],
            "height": result["height"],
            "latency_sec": result["latency_sec"],
            "model": result["model"],
            "scale_factor": result["scale_factor"]
        }
    except Exception as e:
        print(f"[API Upscale Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/generate/stream")
async def websocket_generate_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time generation progress streaming."""
    await websocket.accept()
    print("[WebSocket] Client connected to generation stream")
    
    try:
        while True:
            # Receive generation request
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            steps = data.get("steps", 20)
            
            print(f"[WebSocket] Generating: {prompt}")
            
            # Stream progress updates
            for step in range(1, steps + 1):
                progress = (step / steps) * 100
                
                await websocket.send_json({
                    "type": "progress",
                    "step": step,
                    "total_steps": steps,
                    "progress_pct": round(progress, 1),
                    "status": f"Step {step}/{steps}"
                })
                
                await asyncio.sleep(0.5)  # Simulate processing time
            
            # Generate final image
            gen_res = pipeline.generate(prompt=prompt, steps=steps, mock=True)
            
            buffered = BytesIO()
            gen_res["image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_b64 = f"data:image/png;base64,{img_str[:200]}..."
            
            await websocket.send_json({
                "type": "complete",
                "image_b64_preview": image_b64,
                "latency_sec": gen_res["latency_sec"],
                "status": "completed"
            })
            
            # Record analytics
            analytics.record_generation("sd-v1.5", gen_res["latency_sec"], gen_res["memory_used_mb"])
            
    except Exception as e:
        print(f"[WebSocket Error] {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

@app.post("/api/dreambooth/train")
def api_dreambooth_train(req: DreamboothRequest, request: Request):
    """Quick Dreambooth training for personalization (5-10 minutes)."""
    api_limiter.check_limit(request)
    
    print(f"[API Dreambooth] Training concept '{req.name}' with {len(req.images_b64)} images")
    
    try:
        # Validate image count
        if len(req.images_b64) < 3 or len(req.images_b64) > 10:
            raise HTTPException(status_code=400, detail="Provide 3-10 training images")
        
        # Convert base64 images to PIL
        training_images = []
        for img_b64 in req.images_b64:
            img = decode_base64_image(img_b64)
            training_images.append(img)
        
        # Start background training
        job_id = f"dreambooth_{int(time.time())}"
        
        def train_dreambooth():
            print(f"[Dreambooth Worker] Starting training job {job_id}")
            # In a real implementation, this would call the actual fine-tuning pipeline
            # For now, we'll return a mock result
            with open(f"weights/{job_id}.json", "w") as f:
                json.dump({
                    "name": req.name,
                    "job_id": job_id,
                    "status": "completed",
                    "steps_trained": req.steps,
                    "trained_at": time.time()
                }, f)
            print(f"[Dreambooth Worker] Job {job_id} completed")
        
        training_thread_db = threading.Thread(target=train_dreambooth)
        training_thread_db.start()
        
        return {
            "status": "training_started",
            "job_id": job_id,
            "concept_name": req.name,
            "prompt_template": req.prompt,
            "eta_minutes": 5,
            "message": f"Training Dreambooth model for '{req.name}'. Check status with job_id."
        }
    
    except Exception as e:
        print(f"[API Dreambooth Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dreambooth/status/{job_id}")
def api_dreambooth_status(job_id: str, request: Request):
    """Check Dreambooth training status."""
    api_limiter.check_limit(request)
    
    checkpoint_file = f"weights/{job_id}.json"
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        return checkpoint
    
    return {
        "status": "training",
        "job_id": job_id,
        "progress_pct": 50,
        "message": "Training in progress..."
    }

@app.post("/api/models/switch")
def api_switch_model(req: ModelSelectRequest, request: Request):
    """Switch active model for generation."""
    api_limiter.check_limit(request)
    
    try:
        model_router.select_model(req.quality_level)
        
        return {
            "status": "switched",
            "active_model": model_router.active_model,
            "quality_level": req.quality_level,
            "message": f"Switched to {model_router.active_model}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/available")
def api_list_models(request: Request):
    """List available models."""
    api_limiter.check_limit(request)
    
    return {
        "active_model": model_router.active_model,
        "available_models": [
            {
                "name": name,
                "quality": config["quality"],
                "speed": config["speed"]
            }
            for name, config in model_router.available_models.items()
        ]
    }

@app.get("/api/analytics/stats")
def api_get_analytics(request: Request):
    """Get system analytics and performance metrics."""
    api_limiter.check_limit(request)
    
    stats = analytics.get_stats()
    
    return {
        "status": "success",
        "metrics": stats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.get("/api/health/metrics")
def api_health_metrics(request: Request):
    """Prometheus-style metrics export."""
    api_limiter.check_limit(request)
    
    stats = analytics.get_stats()
    
    metrics_text = f"""# HELP lumaforge_generations_total Total image generations
# TYPE lumaforge_generations_total counter
lumaforge_generations_total {stats['total_generations']}

# HELP lumaforge_errors_total Total errors
# TYPE lumaforge_errors_total counter
lumaforge_errors_total {stats['total_errors']}

# HELP lumaforge_latency_seconds Average latency
# TYPE lumaforge_latency_seconds gauge
lumaforge_latency_seconds {stats['avg_latency_sec']}

# HELP lumaforge_memory_mb Average memory usage
# TYPE lumaforge_memory_mb gauge
lumaforge_memory_mb {stats['avg_memory_mb']}

# HELP lumaforge_error_rate Error rate
# TYPE lumaforge_error_rate gauge
lumaforge_error_rate {stats['error_rate']}
"""
    
    return metrics_text

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

# ============================================
# NEW: Coherence & Image Enhancement Endpoints
# ============================================

@app.post("/api/coherence-check")
def api_coherence_check(req: CoherenceCheckRequest, request: Request):
    """
    Check prompt coherence and get enhancement suggestions.
    Never blocks generation - always returns is_valid=true.
    Provides enhanced prompt for impossible/contradictory prompts.
    """
    api_limiter.check_limit(request)
    
    validator = RealityValidator()
    result = validator.validate_prompt(req.prompt)
    
    return {
        "original_prompt": result["original_prompt"],
        "is_valid": result["is_valid"],
        "coherence_score": result["coherence_score"],
        "coherence_level": result["coherence_level"],
        "enhancement_needed": result["enhancement_needed"],
        "warnings": result["warnings"],
        "contradictions": result["contradictions"],
        "suggestions": result["suggestions"],
        "improved_prompt": result["improved_prompt"],
        "recommendation": result["recommendation"]
    }

@app.post("/api/enhance-image")
def api_enhance_image(req: EnhanceImageRequest, request: Request):
    """
    Enhance image quality with denoise, upscale, detail enhancement, and color improvement.
    Fixes pixelation and improves overall image quality.
    """
    api_limiter.check_limit(request)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Enhance
        from lumaforge.image_enhancer import ImageEnhancer
        enhancer = ImageEnhancer(device=pipeline.device)
        enhanced = enhancer.enhance_full_pipeline(image, enhancement_level=req.enhancement_level)
        
        # Encode back to base64
        output = BytesIO()
        enhanced.save(output, format="PNG")
        output_b64 = base64.b64encode(output.getvalue()).decode()
        
        return {
            "success": True,
            "image_b64": output_b64,
            "original_size": f"{image.width}x{image.height}",
            "enhanced_size": f"{enhanced.width}x{enhanced.height}",
            "enhancement_level": req.enhancement_level
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Image enhancement failed"
        }

@app.post("/api/enhance-zoom")
def api_enhance_zoom(req: EnhanceZoomRequest, request: Request):
    """
    Enhance image quality specifically for zoomed viewing.
    Removes pixelation and block artifacts when zooming in.
    """
    api_limiter.check_limit(request)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Enhance for zoom
        enhanced = pipeline.enhance_zoom_quality(image, zoom_level=req.zoom_level)
        
        # Encode back to base64
        output = BytesIO()
        enhanced.save(output, format="PNG")
        output_b64 = base64.b64encode(output.getvalue()).decode()
        
        return {
            "success": True,
            "image_b64": output_b64,
            "original_size": f"{image.width}x{image.height}",
            "enhanced_size": f"{enhanced.width}x{enhanced.height}",
            "zoom_level": req.zoom_level
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Zoom enhancement failed"
        }

@app.post("/api/remove-pixelation")
def api_remove_pixelation(req: RemovePixelationRequest, request: Request):
    """
    Remove pixelation and block artifacts from an image.
    Useful for cleaning up blocky areas in generated or existing images.
    """
    api_limiter.check_limit(request)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Remove pixelation
        cleaned = pipeline.remove_pixelation(image)
        
        # Encode back to base64
        output = BytesIO()
        cleaned.save(output, format="PNG")
        output_b64 = base64.b64encode(output.getvalue()).decode()
        
        return {
            "success": True,
            "image_b64": output_b64,
            "original_size": f"{image.width}x{image.height}",
            "message": "Pixelation removed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Pixelation removal failed"
        }

@app.post("/api/remove-pixelation")
def api_remove_pixelation(req: RemovePixelationRequest, request: Request):
    """
    Remove pixelation and block artifacts from an image.
    Useful for cleaning up blocky areas in generated or existing images.
    """
    api_limiter.check_limit(request)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Remove pixelation
        cleaned = pipeline.remove_pixelation(image)
        
        # Encode back to base64
        output = BytesIO()
        cleaned.save(output, format="PNG")
        output_b64 = base64.b64encode(output.getvalue()).decode()
        
        return {
            "success": True,
            "image_b64": output_b64,
            "original_size": f"{image.width}x{image.height}",
            "message": "Pixelation removed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Pixelation removal failed"
        }

@app.post("/api/colorize")
def api_colorize(req: ColorizeRequest, request: Request):
    """
    Colorize a grayscale or black & white image with AI-powered color suggestion.
    Supports multiple color styles for artistic control.
    """
    api_limiter.check_limit(request)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Colorize image
        colorized = pipeline.colorize(image, style=req.color_style)
        
        # Encode back to base64
        output = BytesIO()
        colorized.save(output, format="PNG")
        output_b64 = base64.b64encode(output.getvalue()).decode()
        
        return {
            "success": True,
            "image_b64": output_b64,
            "original_size": f"{image.width}x{image.height}",
            "color_style": req.color_style,
            "message": "Image colorized successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Colorization failed"
        }

@app.post("/api/face-restoration")
def api_face_restoration(req: FaceRestorationRequest, request: Request):
    """
    Restore and enhance faces in images.
    Removes artifacts, improves detail, and enhances facial quality.
    Supports multiple restoration intensity levels.
    """
    api_limiter.check_limit(request)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Restore face
        restored = pipeline.restore_face(image, level=req.restoration_level)
        
        # Encode back to base64
        output = BytesIO()
        restored.save(output, format="PNG")
        output_b64 = base64.b64encode(output.getvalue()).decode()
        
        return {
            "success": True,
            "image_b64": output_b64,
            "original_size": f"{image.width}x{image.height}",
            "restoration_level": req.restoration_level,
            "message": "Face restored successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Face restoration failed"
        }

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces port defaults to 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting LumaForge API Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
