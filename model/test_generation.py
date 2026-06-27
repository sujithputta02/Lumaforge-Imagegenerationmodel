#!/usr/bin/env python3
"""Test SDXL Turbo image generation"""
import requests
import time
from PIL import Image
import io
import numpy as np

# Test the wizard prompt
prompt = "a wizard with a long white beard standing in a mystical forest"
print(f"🧙 Testing SDXL Turbo with prompt: '{prompt}'")
print("")

# Start generation session
print("Starting generation session...")
start_response = requests.post("http://localhost:7860/api/generate-session/start", json={
    "prompt": prompt,
    "mode": "general",
    "aspect_ratio": "1:1",
    "steps": 4,
    "guidance_scale": 0.0,
    "seed": -1,
    "mock": False
})

if start_response.status_code == 200:
    session_data = start_response.json()
    session_id = session_data["session_id"]
    print(f"✅ Session started: {session_id}")
    print("")
    
    # Poll for completion
    print("⏳ Generating image", end="", flush=True)
    while True:
        status_response = requests.post("http://localhost:7860/api/generate-session/status", json={
            "session_id": session_id
        })
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            state = status_data["state"]
            
            if state == "completed":
                print(" ✅")
                print("")
                print("Generation completed!")
                print(f"  Image URL: {status_data['image_url']}")
                print(f"  Time: {status_data['latency_sec']:.1f}s")
                print(f"  Memory: {status_data['memory_used_mb']:.1f}MB")
                print(f"  Seed: {status_data['seed']}")
                print(f"  Mock: {status_data['used_mock']}")
                print("")
                
                # Check if image is not blank
                img_response = requests.get(f"http://localhost:7860{status_data['image_url']}")
                if img_response.status_code == 200:
                    img = Image.open(io.BytesIO(img_response.content))
                    img_array = np.array(img)
                    
                    # Check if image is blank (all black or all same color)
                    is_blank = (img_array.std() < 5)
                    mean_brightness = img_array.mean()
                    
                    if is_blank:
                        print("❌ WARNING: Image appears to be BLANK/BLACK!")
                        print(f"   Mean brightness: {mean_brightness:.1f}/255")
                        print(f"   Std deviation: {img_array.std():.1f}")
                        print("")
                        print("The upcast_vae fix may not have worked. Check backend logs.")
                    else:
                        print("✅ SUCCESS! Image looks good (Not blank)")
                        print(f"   Mean brightness: {mean_brightness:.1f}/255")
                        print(f"   Std deviation: {img_array.std():.1f}")
                        print(f"   Image size: {img.size}")
                        print("")
                        print(f"🎨 View your image at: http://localhost:3000")
                    
                break
            elif state == "failed":
                print(" ❌")
                print(f"Generation failed: {status_data.get('error', 'Unknown error')}")
                break
            elif state == "generating":
                print(".", end="", flush=True)
                time.sleep(1)
        else:
            print(f"Status check failed: {status_response.status_code}")
            break
else:
    print(f"❌ Failed to start session: {start_response.status_code}")
    print(start_response.text)
