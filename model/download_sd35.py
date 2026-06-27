#!/usr/bin/env python3
"""Download Stable Diffusion 3.5 Medium for high-quality inference"""
from diffusers import StableDiffusion3Pipeline
import torch
import os

print("🚀 Downloading Stable Diffusion 3.5 Medium...")
print("📦 Size: ~5-6GB")
print("🎨 Latest Stability AI model with excellent quality!")
print("")

model_id = "stabilityai/stable-diffusion-3.5-medium"
token = os.getenv("HF_TOKEN")

# Expand cache dir properly
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print("⬇️  Downloading SD 3.5 Medium with authentication...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    token=token,
    resume_download=True
)

print("")
print("✅ SD 3.5 Medium downloaded successfully!")
print(f"💾 Cached at: {cache_dir}")
print("")
print("🎯 Next steps:")
print("   1. Restart backend: cd model && python3 app.py")
print("   2. Test at: http://localhost:3000")
print("   3. Expected: Best quality, 25-35 seconds!")
