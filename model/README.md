---
title: LumaForge-Image Generation Model
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🌌 LumaForge AuraGen: Latent Diffusion & Fine-Tuning API Engine

This is the self-contained backend API engine for LumaForge AuraGen, designed to be deployed directly to **Hugging Face Spaces** (Docker or Python API spaces). It provides endpoints for Text-to-Image generation, Image-to-Image styling, 2x upscaling, background removal, dataset curation, and LoRA fine-tuning telemetry.

---

## 🚀 Hugging Face Space Deployment

Hugging Face Spaces automatically detect configuration metadata from the YAML frontmatter at the top of this file.

### Option A: Docker Space (Recommended)
This folder is configured to run on port `7860` (the default Hugging Face Space port). You can create a Hugging Face space using the **Docker** SDK and push the contents of the `model/` directory along with a standard `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Pillow and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Run FastAPI server
CMD ["python", "app.py"]
```

### Option B: FastAPI Space
Create a Hugging Face space with the `FastAPI` SDK, selecting **Python 3.10**, and copy the contents of the `model/` directory. Hugging Face will automatically recognize `app.py` as the entrypoint.

---

## 📡 API Endpoints Reference

### 1. System Status
* **`GET /api/status`**
  * Returns device specs (Metal MPS vs CPU) and local Ollama server connectivity logs.

### 2. Text-to-Image Generation
* **`POST /api/generate`**
  * **Payload**:
    ```json
    {
      "prompt": "studio ghibli street",
      "mode": "general | poster | character",
      "aspect_ratio": "1:1 | 16:9 | 9:16 | 4:3 | 3:4",
      "steps": 20,
      "guidance_scale": 7.5,
      "seed": -1,
      "mock": false
    }
    ```
  * **Actions**: Checks text safety boundaries (Ollama client),适配 expands prompts structurally, runs latent diffusion on MPS, watermarks the result with the LumaForge logo, and returns the image as a Base64 string.

### 3. Image-to-Image Stylization
* **`POST /api/generate-img2img`**
  * **Payload**:
    ```json
    {
      "prompt": "Convert this photo into anime illustration",
      "image_b64": "data:image/png;base64,...",
      "strength": 0.32,
      "mode": "general",
      "steps": 20,
      "guidance_scale": 7.5,
      "seed": -1,
      "mock": false
    }
    ```
  * **Actions**: Styles the input image using shared pipeline weights. Caps strength to `0.32` and applies a **Radial Face Protection Mask** to preserve original facial structure and details with pixel-level accuracy.

### 4. High-Fidelity 2x Upscaling
* **`POST /api/upscale`**
  * **Payload**:
    ```json
    {
      "image_b64": "data:image/png;base64,...",
      "scale_factor": 2.0,
      "mock": false
    }
    ```
  * **Actions**: Doubles the resolution of the image using high-quality Lanczos interpolation and sharpens details using an Unsharp Mask.

### 5. Transparent Background Removal
* **`POST /api/remove-background`**
  * **Payload**:
    ```json
    {
      "image_b64": "data:image/png;base64,...",
      "mock": false
    }
    ```
  * **Actions**: Isolates the foreground subject. Uses `rembg` if available, falling back to a vectorized NumPy color-threshold algorithm featuring linear alpha feathering to prevent jagged edges.

### 6. Model Training Telemetry
* **`POST /api/train`**: Triggers PyTorch UNet LoRA layer fine-tuning on a background thread.
* **`GET /api/train/status`**: Returns live telemetry logs (epoch progress, validation loss metrics, prompt adherence).

### 7. Dataset Curation & Benchmarking
* **`POST /api/curate`**: Curates and captions images.
* **`POST /api/benchmark`**: Evaluates pipeline adherence, processing latency, and VRAM footprints.

---

## ⚡ Performance Optimizations
* **Attention Slicing**: Pipeline memory slicing allows Stable Diffusion to run on standard consumer MPS buffers without out-of-memory errors.
* **Vectorized Processing**: Replaced slow pixel iteration loops with fast vectorized NumPy operations, reducing processing latencies (Sketch generation to **4ms**, Background removal to **8ms**).
* **Token-Bucket Rate Limiters**: Restricts API calls to prevent client flooding (10 generations/min, 60 general api calls/min).
