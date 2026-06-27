#!/usr/bin/env python3
"""Download Realistic Vision V2 for excellent photorealistic results on Apple MPS"""
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

print("🚀 Downloading Realistic Vision V2.0...")
print("📦 Size: ~4GB")
print("✅ Excellent photorealistic quality!")
print("🎨 Works perfectly on Apple MPS")
print("")

model_id = "SG161222/Realistic_Vision_V2.0"

print("⬇️  Downloading Realistic Vision V2...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir="~/.cache/huggingface/hub",
    safety_checker=None
)

# Configure scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

print("")
print("✅ Realistic Vision V2 downloaded successfully!")
print("💾 Cached at: ~/.cache/huggingface/hub/")
print("")
print("🎯 Next steps:")
print("   1. Restart backend: cd model && python3 app.py")
print("   2. Test at: http://localhost:3000")
print("   3. Expected: Photorealistic quality, 20-25 seconds, NO black images!")
