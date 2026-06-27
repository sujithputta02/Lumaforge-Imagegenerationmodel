#!/usr/bin/env python3
"""Download SDXL Turbo fp16 variant (7GB) for faster performance"""
from diffusers import AutoPipelineForText2Image
import torch
import os

print("🚀 Downloading SDXL Turbo fp16 variant...")
print("📦 Size: ~7GB (much faster than float32)")
print("")

model_id = "stabilityai/sdxl-turbo"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print("⬇️  Downloading fp16 variant...")
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=cache_dir,
    resume_download=True  # Resume if interrupted
)

print("")
print("✅ SDXL Turbo fp16 downloaded successfully!")
print("💾 Cached at: ~/.cache/huggingface/hub/")
print("")
print("🎯 Next steps:")
print("   1. Restart backend: cd model && python3 app.py")
print("   2. Test at: http://localhost:3000")
print("   3. Expected: Fast inference, NO black images!")
