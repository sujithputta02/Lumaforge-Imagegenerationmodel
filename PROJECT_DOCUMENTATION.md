# LumaForge - Technical Project Documentation

## Executive Summary

LumaForge (originally AuraGen MPS) is a local-first, AI-powered text-to-image generation platform optimized for Apple Silicon. It combines state-of-the-art diffusion models with intelligent prompt enhancement, safety moderation, and a comprehensive web-based interface for creative image generation. The system is designed to run entirely on-device using Metal Performance Shaders (MPS), ensuring privacy, low latency, and offline capability.

## Project Architecture

### High-Level Overview

LumaForge follows a modular architecture with three main components:

1. **Backend API Server** - FastAPI-based REST API handling generation requests
2. **Core Pipeline Engine** - Stable Diffusion integration with custom enhancements
3. **Frontend Web Application** - Next.js-based interactive playground

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Frontend (Next.js)                   │
│  - Interactive Playground                                    │
│  - Real-time Generation Preview                              │
│  - Advanced Effects & Editing Tools                          │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST API
┌────────────────────┴────────────────────────────────────────┐
│                 Backend API (FastAPI)                        │
│  - Request Validation & Rate Limiting                        │
│  - Session Management                                        │
│  - Background Task Queue                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Core Pipeline (LumaForgePipeline)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Safety Manager (Ollama-based)                      │   │
│  │   - Prompt Classification                            │   │
│  │   - Content Moderation                               │   │
│  │   - Semantic Rewriting                               │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Ollama Client                                      │   │
│  │   - Prompt Expansion & Enhancement                   │   │
│  │   - SD 3.5 Optimization                              │   │
│  │   - Coherence Checking                               │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Stable Diffusion Pipeline                          │   │
│  │   - Text-to-Image Generation                         │   │
│  │   - Image-to-Image Transformation                    │   │
│  │   - Upscaling & Enhancement                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. LumaForgePipeline (`lumaforge/pipeline.py`)


The central pipeline class responsible for all image generation operations.

**Key Features:**
- **Multi-Model Support**: Compatible with SD 2.1, SD 3.5 Medium, SDXL Turbo
- **Text-to-Image**: Generate images from natural language prompts
- **Image-to-Image**: Transform existing images based on prompts
- **Intelligent Upscaling**: 2x/4x upscaling with face restoration
- **Background Removal**: AI-powered subject isolation
- **Colorization**: Multiple style presets (vibrant, warm, cool, vintage, sepia)
- **Face Restoration**: Multi-level face enhancement (low, medium, high, ultra)
- **Advanced Effects**: Depth-of-field, film grain, chromatic aberration, lens flare

**Technical Details:**
- Uses PyTorch with MPS (Metal Performance Shaders) acceleration
- Supports multiple aspect ratios: 1:1, 16:9, 9:16, 4:3, 3:4
- Configurable inference steps (optimized for SD 3.5: 28 steps, guidance 4.5)
- Memory-aware generation with automatic resource management
- Mock mode for testing without GPU usage

**Generation Modes:**
- `general`: Standard creative generation
- `poster`: Movie poster layouts with typography space
- `character`: Character-focused compositions

### 2. Safety Manager (`lumaforge/safety.py`)


Multi-layered content moderation system ensuring safe image generation.

**Safety Pipeline:**
1. **Input Classification**: Categorizes prompts as SAFE, BORDERLINE, or UNSAFE
2. **Semantic Rewriting**: Transforms borderline prompts into safe alternatives
3. **Audit Logging**: JSONL-based event logging for compliance
4. **Output Validation**: Post-generation safety checks

**Classification Results:**
- `APPROVED`: Safe prompt, proceeds unchanged
- `REWRITTEN`: Borderline prompt transformed to safe alternative
- `REFUSED`: Unsafe content blocked completely

**Audit Trail:**
Each moderation event logs:
- Timestamp
- Event type (INPUT_PROMPT, OUTPUT_IMAGE)
- Original and processed prompts
- Classification and reasoning
- Status and latency metrics

### 3. Ollama Client (`lumaforge/ollama_client.py`)

LLM-powered prompt intelligence layer using local Ollama models.

**Core Functions:**

**a) Prompt Expansion:**
Transforms brief prompts into detailed, structured descriptions with:
- Subject details
- Style attributes
- Lighting descriptions
- Camera angles
- Mood and atmosphere

**b) SD 3.5 Optimization:**
- Token limit enforcement (256 tokens max)
- Natural language compression
- Quality enhancement terms
- Coherent narrative structure

**c) Coherence Checking:**
Analyzes prompts for:
- Logical consistency
- Conflicting elements
- Generative feasibility
- Structural issues

**d) Safety Classification:**
- Pattern matching for unsafe content
- Context-aware evaluation
- Fallback heuristics when Ollama unavailable

**Fallback Mode:**
When Ollama is offline, uses heuristic classifiers to maintain functionality.

### 4. Dataset Curator (`lumaforge/dataset_curator.py`)

Automated dataset preparation and curation system for fine-tuning.

**Features:**
- **Image Collection**: Downloads diverse creative images
- **Quality Validation**: Checks image integrity and format
- **Deduplication**: Perceptual hash-based duplicate detection
- **AI Captioning**: Ollama-generated descriptive captions
- **JSONL Metadata**: Structured prompt-image pairs

**Curation Pipeline:**
1. Download images from diverse sources
2. Validate image format and dimensions
3. Calculate perceptual hashes
4. Filter duplicates (>95% similarity threshold)
5. Generate descriptive captions via Ollama
6. Save to `data/creative_dataset/` with metadata.jsonl

**Output Format:**
```json
{
  "image_path": "images/image_000.jpg",
  "caption": "A cinematic portrait of a futuristic warrior...",
  "source": "generated",
  "category": "character"
}
```

### 5. LumaForge Trainer (`lumaforge/train.py`)

Fine-tuning system for model customization and style adaptation.

**Training Features:**
- **LoRA-based Fine-tuning**: Efficient parameter adaptation
- **Style Weight Balancing**: Prevents dataset bias
- **Demo Mode**: Fast validation with synthetic data
- **Checkpoint Management**: Resume from interruptions
- **Cooldown Periods**: Memory pressure management
- **Validation Metrics**: Prompt adherence, loss tracking

**Training Process:**
1. Load curated dataset
2. Calculate style distribution weights
3. Initialize/resume from checkpoint
4. Train with validation splits
5. Monitor prompt adherence and loss
6. Save checkpoints at intervals
7. Export final LoRA weights

**Metrics Tracked:**
- Training loss
- Validation loss
- Prompt adherence score
- Style distribution balance
- Memory usage
- Epoch progress

### 6. Benchmark Suite (`lumaforge/benchmark.py`)

Automated evaluation framework for quality assurance.

**Test Categories:**
- Single Subject: Portrait and object generation
- Multi-Character: Complex scene composition
- Movie Poster: Layout and typography handling
- Text-in-Image: Text rendering accuracy
- Safety Boundary: Moderation effectiveness

**Evaluation Metrics:**
- Prompt adherence score
- Generation latency (seconds)
- Memory consumption (MB)
- Safety refusal precision/recall
- Image quality ratings

**Output:**
Generates `benchmark_report.json` with:
- Summary statistics
- Per-test results
- Safety classification accuracy
- Performance metrics
- Saved output images in `benchmark_outputs/`

## API Server (`model/app.py`)

FastAPI-based REST API providing comprehensive endpoints for all features.

### Core Endpoints

#### Generation Endpoints

**`POST /api/generate`** - Text-to-Image Generation
```python
{
  "prompt": "cyberpunk warrior in neon city",
  "mode": "general",
  "aspect_ratio": "16:9",
  "steps": 28,
  "guidance_scale": 4.5,
  "seed": -1,
  "negative_prompt": "",
  "mock": false
}
```


**`POST /api/generate-img2img`** - Image-to-Image Transformation
```python
{
  "image_b64": "base64_encoded_image",
  "prompt": "transform to anime style",
  "strength": 0.5,
  "steps": 20,
  "guidance_scale": 7.5
}
```

**`POST /api/upscale`** - Image Upscaling
```python
{
  "image_b64": "base64_encoded_image",
  "scale_factor": 2.0,
  "mock": false
}
```

**`POST /api/remove-background`** - Background Removal
```python
{
  "image_b64": "base64_encoded_image",
  "mock": false
}
```

**`POST /api/colorize`** - Image Colorization
```python
{
  "image_b64": "base64_encoded_image",
  "color_style": "vibrant"
}
```

**`POST /api/face-restoration`** - Face Enhancement
```python
{
  "image_b64": "base64_encoded_image",
  "restoration_level": "high"
}
```


#### Session Management (Background Generation)

**`POST /api/generate-session/start`** - Start Background Generation
Returns `session_id` for tracking long-running generations

**`POST /api/generate-session/status`** - Check Session Status
```python
{
  "session_id": "uuid-string"
}
```

**`POST /api/generate-session/cancel`** - Cancel Running Generation

**`POST /api/generate-session/cleanup`** - Clean Up Completed Session

#### Advanced Features

**`POST /api/coherence-check`** - Validate Prompt Quality
```python
{
  "prompt": "a flying submarine in space"
}
```

**`POST /api/enhance-image`** - Remove Artifacts
```python
{
  "image_b64": "base64_encoded_image",
  "enhancement_level": "high"
}
```

**`POST /api/enhance-zoom`** - Zoom Quality Enhancement
```python
{
  "image_b64": "base64_encoded_image",
  "zoom_level": 2
}
```


**`POST /api/enhance/effects`** - Apply Visual Effects
```python
{
  "image_b64": "base64_encoded_image",
  "effect_type": "depth-of-field",
  "intensity": 0.5,
  "params": {
    "focus_point": [0.5, 0.5],
    "blur_strength": 12
  }
}
```

**`POST /api/batch/generate`** - Batch Generation
```python
{
  "prompts": ["prompt1", "prompt2", "prompt3"],
  "count": 3,
  "steps": 28
}
```

#### Training & Evaluation

**`POST /api/train`** - Start Fine-tuning
```python
{
  "epochs": 3,
  "lr": 5e-6,
  "batch_size": 2,
  "demo": true,
  "checkpoint_steps": 100
}
```

**`GET /api/train/status`** - Training Progress

**`POST /api/curate`** - Dataset Curation
```python
{
  "limit": 90,
  "use_ollama": true
}
```


**`POST /api/benchmark`** - Run Evaluation Suite
```python
{
  "mock": true,
  "device": "mps"
}
```

#### Monitoring & Analytics

**`GET /api/status`** - Server Health Check
Returns:
- Server status
- Device (mps/cuda/cpu)
- MPS availability
- Ollama connection status

**`GET /api/audit-log?limit=25`** - Moderation History

**`GET /api/analytics/stats`** - Usage Statistics

**`GET /api/models`** - Available Models List

**`POST /api/models/switch`** - Switch Active Model

### API Features

**Rate Limiting:**
- Request-based limits per endpoint
- Sliding window implementation
- 429 responses with retry information

**Session Management:**
- UUID-based session tracking
- Background worker threads
- Automatic cleanup of stale sessions
- Status polling support

**CORS Configuration:**
- Allows web frontend communication
- Configurable origins
- Credentials support

## Web Frontend (`web/src/app/page.tsx`)

Modern Next.js 16 application with comprehensive UI for all features.

### Features

#### 1. Playground Tab
- Text-to-Image generation interface
- Image-to-Image transformations
- Real-time parameter controls:
  - Aspect ratio selection
  - Step count slider (optimized for SD 3.5: 28 steps)
  - Guidance scale (CFG)
  - Seed control
  - Negative prompts
  - Mock mode toggle

#### 2. Editor Tab
- Inpainting with brush-based masking
- Outpainting for image expansion
- Canvas-based editing tools
- Task presets (addition, removal, style transfer, etc.)

#### 3. Effects Tab
- Depth-of-field with focus control
- Film grain simulation
- Chromatic aberration
- Lens flare effects
- Real-time parameter adjustment

#### 4. Batch Generation Tab
- Multiple prompt processing
- Parallel generation support
- Batch result gallery

#### 5. Training Tab
- Fine-tuning parameter configuration
- Real-time training telemetry
- Progress tracking with metrics:
  - Epoch progress
  - Training/validation loss
  - Prompt adherence scores
- Training history visualization

#### 6. Dreambooth Tab
- Custom concept training
- Multiple image upload
- Unique token configuration
- Training status monitoring

#### 7. Models Tab
- Available models listing
- Model switching interface
- Model specifications display
- Performance metrics

#### 8. Audit Log Tab
- Moderation event history
- Timestamp tracking
- Classification results
- Original vs processed prompts


#### 9. Benchmark Tab
- Evaluation suite execution
- Performance report viewing
- Quality metrics display

#### 10. Analytics Tab
- Usage statistics
- Performance trends
- Generation analytics

### UI/UX Features

**Real-time Feedback:**
- Generation progress indicators
- Stage-by-stage updates
- Session-based background processing
- Cancellation support

**Image Operations:**
- Drag-and-drop upload
- One-click download
- Upscale button
- Background removal
- Colorization with style presets
- Face restoration with intensity levels

**Metadata Display:**
- Generation parameters
- Latency metrics
- Memory usage
- Seed values
- Prompt expansion details
- Safety check results

**Health Monitoring:**
- Server status indicator
- Device information
- Ollama connection status
- MPS availability


### Technology Stack

**Frontend:**
- Next.js 16.2.9
- React 19.2.4
- TypeScript 5
- Tailwind CSS 4
- Lucide React (icons)

**Backend:**
- FastAPI
- Uvicorn (ASGI server)
- Pydantic v2 (validation)

**AI/ML:**
- PyTorch 2.0+
- Diffusers 0.19+
- Transformers 4.30+
- Accelerate 0.20+
- Pillow (image processing)

**LLM Integration:**
- Ollama (local LLM orchestration)
- llama3.2:1b (default model)

## CLI Tool (`main.py`)

Command-line interface for direct pipeline access.

### Commands

**Generate Image:**
```bash
python main.py generate \
  --prompt "cyberpunk warrior" \
  --mode poster \
  --aspect-ratio 16:9 \
  --steps 28 \
  --device mps
```


**Curate Dataset:**
```bash
python main.py curate \
  --limit 90 \
  --use-ollama
```

**Run Benchmark:**
```bash
python main.py benchmark \
  --mock \
  --device mps
```

**Fine-tune Model:**
```bash
python main.py train \
  --epochs 5 \
  --lr 5e-6 \
  --batch-size 4 \
  --demo
```

**View Audit Log:**
```bash
python main.py audit-log --limit 50
```

## Project Structure

```
LumaForge/
├── model/                          # Backend AI model system
│   ├── lumaforge/                  # Core package
│   │   ├── __init__.py            # Package exports
│   │   ├── pipeline.py            # Main generation pipeline
│   │   ├── safety.py              # Content moderation
│   │   ├── ollama_client.py       # LLM integration
│   │   ├── dataset_curator.py     # Data preparation
│   │   ├── train.py               # Fine-tuning system
│   │   ├── benchmark.py           # Evaluation suite
│   │   └── category_prompts.py    # Prompt templates
│   ├── app.py                     # FastAPI server
│   ├── download_sd21.py           # Model downloader
│   ├── download_sd35.py           # SD 3.5 downloader
│   ├── download_sdxl_turbo_fp16.py# SDXL Turbo downloader
│   ├── requirements.txt           # Python dependencies
│   ├── audit_log.jsonl            # Moderation log
│   └── outputs/                   # Generated images
├── web/                           # Frontend application
│   ├── src/
│   │   └── app/
│   │       └── page.tsx           # Main UI component
│   ├── public/                    # Static assets
│   ├── package.json               # Node dependencies
│   ├── tsconfig.json              # TypeScript config
│   ├── next.config.ts             # Next.js config
│   └── tailwind.config.ts         # Tailwind config
├── data/
│   └── creative_dataset/          # Training data
│       ├── images/                # Image files
│       └── metadata.jsonl         # Captions
├── weights/
│   └── checkpoints/               # Training checkpoints
├── benchmark_outputs/             # Evaluation results
├── main.py                        # CLI entry point
├── benchmark_report.json          # Latest benchmark
├── train_log.json                 # Training telemetry
├── audit_log.jsonl                # Root audit log
└── PRD.txt                        # Product requirements
```

## Key Technologies & Concepts

### Metal Performance Shaders (MPS)

Apple's GPU acceleration framework for PyTorch on Apple Silicon. Enables:
- On-device neural network inference
- Unified memory architecture utilization
- Efficient tensor operations
- Privacy-preserving local computation

### Stable Diffusion 3.5 Medium

Current default model with optimal parameters:
- **Steps**: 28 (sweet spot for quality/speed)
- **Guidance Scale**: 4.5 (recommended CFG)
- **Architecture**: Latent diffusion with enhanced text encoding
- **Strengths**: Improved prompt adherence, better composition

### LoRA (Low-Rank Adaptation)

Efficient fine-tuning technique used in training:
- Parameter-efficient adaptation
- Maintains base model quality
- Fast training convergence
- Small checkpoint sizes
- Multiple LoRA composability

### Ollama Integration

Local LLM server providing:
- Zero-cost prompt enhancement
- Privacy-preserving moderation
- Offline functionality
- Extensible model support
- Fast inference on Apple Silicon

## Workflow Examples

### Standard Generation Flow

1. User enters prompt in web UI
2. Frontend sends request to `/api/generate-session/start`
3. Backend creates session, starts background worker
4. Safety Manager classifies prompt
5. If BORDERLINE, Ollama rewrites prompt
6. If UNSAFE, generation refused immediately
7. Ollama expands prompt for better quality
8. Pipeline generates image with SD 3.5
9. Safety check on output image
10. Result encoded to base64
11. Session updated with result
12. Frontend polls and displays image

### Image-to-Image Flow

1. User uploads image and provides transformation prompt
2. Frontend sends base64 image + prompt to `/api/generate-img2img`
3. Backend decodes image
4. Safety check on input prompt
5. Ollama enhances transformation prompt
6. Pipeline applies img2img with strength parameter
7. Generated image returned
8. Optional post-processing (upscale, enhance)

### Training Flow

1. User curates dataset via `/api/curate`
2. Curator downloads/validates images
3. Ollama generates descriptive captions
4. Metadata saved to JSONL
5. User configures training params in UI
6. Frontend calls `/api/train`
7. Background worker starts training
8. Frontend polls `/api/train/status` for progress
9. Training logs saved with metrics
10. Checkpoints saved at intervals
11. Final LoRA weights exported


## Performance Characteristics

### Generation Speed

**SD 3.5 Medium on M1/M2 Macs:**
- 1024x1024: ~8-15 seconds @ 28 steps
- 1920x1080: ~10-20 seconds @ 28 steps
- Memory: ~6-8 GB unified memory

**Factors Affecting Performance:**
- Image resolution
- Inference steps
- Model complexity
- Memory pressure
- Background processes

### Optimization Strategies

**Memory Management:**
- Automatic garbage collection
- Model unloading when idle
- Batch size tuning
- Gradient checkpointing (training)

**Speed Optimization:**
- Optimal step counts (28 for SD 3.5)
- Cached embeddings
- Mock mode for testing
- Progressive previews

**Quality Optimization:**
- Prompt expansion via Ollama
- Negative prompts
- Guidance scale tuning
- Multi-stage enhancement

## Safety & Moderation

### Three-Layer Safety System


**Layer 1: Input Classification**
- Pattern matching for explicit terms
- Context analysis via Ollama
- Classification: SAFE, BORDERLINE, UNSAFE

**Layer 2: Semantic Rewriting**
- Transforms borderline prompts
- Preserves creative intent
- Removes problematic elements
- Maintains generative quality

**Layer 3: Output Validation**
- Post-generation safety check
- Image content analysis
- Audit trail logging

### Audit Trail

Every generation logs:
- Original prompt
- Final prompt (after rewriting)
- Classification result
- Safety decision
- Timestamp
- Latency metrics

**Compliance Features:**
- JSONL format for parsing
- Immutable append-only log
- Privacy-preserving (local only)
- Searchable history

## Use Cases

### 1. Creative Professionals
- **Concept Art**: Rapid ideation for games, films, products
- **Storyboarding**: Visual scene planning
- **Mood Boards**: Style exploration
- **Client Presentations**: Quick mockups


### 2. Content Creators
- **Social Media**: Custom graphics and visuals
- **YouTube Thumbnails**: Eye-catching designs
- **Blog Headers**: Article illustrations
- **Marketing Materials**: Ad creatives

### 3. Developers & Startups
- **App Mockups**: UI/UX visualization
- **Product Images**: Hero shots for landing pages
- **Game Assets**: Character and environment concepts
- **Embedded Generation**: SDK integration

### 4. Educators & Students
- **Learning Tool**: Understanding AI art generation
- **Research Platform**: Experimenting with models
- **Portfolio Projects**: Demonstrating capabilities
- **Safe Environment**: Moderated content generation

### 5. Privacy-Conscious Users
- **Local Processing**: No cloud upload required
- **Offline Mode**: Works without internet
- **Data Control**: All assets stay on device
- **No Tracking**: No external analytics

## Installation & Setup

### Prerequisites

**System Requirements:**
- macOS with Apple Silicon (M1/M2/M3)
- 16GB+ RAM recommended
- 20GB+ free disk space
- Python 3.8+
- Node.js 20+


**Software Dependencies:**
- Ollama (for LLM features)
- Git (for model downloads)

### Backend Setup

```bash
# Navigate to model directory
cd model

# Install Python dependencies
pip install -r requirements.txt

# Download default model (SD 3.5 Medium)
python download_sd35.py

# Install and start Ollama
# Visit: https://ollama.ai
ollama pull llama3.2:1b

# Start API server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
# Navigate to web directory
cd web

# Install dependencies
npm install
# or with bun
bun install

# Start development server
npm run dev
# or
bun dev

# Access at http://localhost:3000
```

### CLI Usage

```bash
# From project root
python main.py generate --prompt "your prompt here"
```

## Configuration


### Model Configuration

**Default Model:**
- `stabilityai/stable-diffusion-3.5-medium`

**Alternative Models:**
- `stabilityai/stable-diffusion-2-1`
- `stabilityai/sdxl-turbo` (faster, lower quality)

**Switching Models:**
```python
# In pipeline.py or via API
pipeline = LumaForgePipeline(
    model_id="stabilityai/stable-diffusion-2-1",
    device="mps"
)
```

### Ollama Configuration

**Default Model:**
- `llama3.2:1b` (fast, efficient)

**Alternative Models:**
```bash
# Install larger model for better quality
ollama pull llama3.2:3b

# Update in ollama_client.py
client = OllamaClient(model="llama3.2:3b")
```

### API Configuration

**Port & Host:**
Edit `model/app.py`:
```python
# Change port or host
uvicorn.run(app, host="0.0.0.0", port=8080)
```

**CORS Origins:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"]
)
```


**Rate Limiting:**
```python
# Adjust limits in app.py
upscale_limiter = RateLimiter(limit=10, window=60)  # 10 req/min
```

## Troubleshooting

### Common Issues

**1. MPS Not Available**
- Ensure running on Apple Silicon Mac
- Update to latest macOS
- Reinstall PyTorch with MPS support

**2. Out of Memory**
- Reduce image resolution
- Lower batch size (training)
- Close other applications
- Use smaller model (SD 2.1)

**3. Ollama Connection Failed**
- Start Ollama service: `ollama serve`
- Check if model installed: `ollama list`
- Verify port 11434 not blocked
- Fallback heuristics activate automatically

**4. Slow Generation**
- Reduce step count (minimum 20)
- Lower resolution
- Use SDXL Turbo for speed
- Check background processes

**5. Poor Image Quality**
- Increase steps (28 recommended for SD 3.5)
- Use prompt expansion (enable Ollama)
- Adjust guidance scale (4.5 for SD 3.5)
- Add negative prompts
- Try different seeds


## Advanced Topics

### Custom LoRA Training

**Prepare Dataset:**
1. Collect 50-100 high-quality images
2. Ensure visual consistency
3. Place in `data/creative_dataset/images/`
4. Run curator for captions

**Train LoRA:**
```bash
python main.py train \
  --epochs 10 \
  --lr 1e-5 \
  --batch-size 2 \
  --checkpoint-steps 50
```

**Apply LoRA:**
```python
# In pipeline.py
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("base-model")
pipe.load_lora_weights("weights/checkpoints/final")
```

### Extending Safety Rules

**Add Custom Patterns:**
Edit `lumaforge/safety.py`:
```python
unsafe_patterns = [
    "violence", "gore", "explicit",
    "your-custom-pattern"
]
```

**Custom Ollama Prompts:**
Edit classification prompt in `ollama_client.py`:
```python
system_prompt = """
You are a safety classifier...
Additional rule: [your rule]
"""
```

### Creating Category Prompts


Edit `lumaforge/category_prompts.py`:
```python
CATEGORY_PROMPTS = {
    "fantasy": {
        "dragon": "majestic dragon, epic scale...",
        "wizard": "powerful wizard, mystical aura..."
    },
    "scifi": {
        "spaceship": "advanced spaceship, futuristic design...",
        "robot": "humanoid robot, advanced AI..."
    }
}
```

### API Integration

**Python Client Example:**
```python
import requests
import base64

def generate_image(prompt):
    response = requests.post(
        "http://localhost:8000/api/generate",
        json={
            "prompt": prompt,
            "steps": 28,
            "guidance_scale": 4.5,
            "aspect_ratio": "1:1"
        }
    )
    result = response.json()
    
    # Decode base64 image
    image_data = base64.b64decode(
        result["image_b64"].split(",")[1]
    )
    
    with open("output.png", "wb") as f:
        f.write(image_data)
    
    return result

# Usage
result = generate_image("cyberpunk city at night")
print(f"Generated in {result['generation_metadata']['latency_sec']}s")
```


**JavaScript/TypeScript Client:**
```typescript
async function generateImage(prompt: string) {
    const response = await fetch('http://localhost:8000/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt,
            steps: 28,
            guidance_scale: 4.5,
            aspect_ratio: '16:9'
        })
    });
    
    const result = await response.json();
    
    // Display image
    const img = document.createElement('img');
    img.src = result.image_b64;
    document.body.appendChild(img);
    
    return result;
}

// Usage
generateImage('futuristic cityscape')
    .then(r => console.log(`Seed: ${r.generation_metadata.seed}`));
```

### Batch Processing Script

```python
# batch_generate.py
from lumaforge import LumaForgePipeline, SafetyManager, OllamaClient
import json

# Initialize
ollama = OllamaClient()
safety = SafetyManager(ollama_client=ollama)
pipeline = LumaForgePipeline(ollama_client=ollama)

# Load prompts
with open('prompts.txt', 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

results = []
for i, prompt in enumerate(prompts):
    print(f"Processing {i+1}/{len(prompts)}: {prompt}")
    
    # Safety check
    mod_result = safety.moderate_prompt(prompt)
    if mod_result['status'] == 'REFUSED':
        print(f"  ❌ Refused: {mod_result['reason']}")
        continue
    
    # Generate
    gen_result = pipeline.generate(
        prompt=mod_result['final_prompt'],
        aspect_ratio='16:9',
        steps=28
    )
    
    # Save
    output_path = f"outputs/batch_{i:03d}.png"
    gen_result['image'].save(output_path)
    
    results.append({
        'prompt': prompt,
        'output': output_path,
        'seed': gen_result['seed'],
        'latency': gen_result['latency_sec']
    })
    
    print(f"  ✅ Saved to {output_path}")

# Save manifest
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Development Guidelines

### Code Organization

**Module Responsibilities:**
- `pipeline.py`: Image generation core, no business logic
- `safety.py`: Moderation only, no generation code
- `ollama_client.py`: LLM interaction, no image handling
- `app.py`: API routing and validation only
- `train.py`: Training logic, separate from inference

**Separation of Concerns:**
- Keep API routes thin, delegate to modules
- No direct model calls in frontend
- Use dependency injection for clients
- Centralize configuration

### Testing Approach

**Unit Tests:**
```python
# test_safety.py
def test_safe_prompt_approved():
    safety = SafetyManager()
    result = safety.moderate_prompt("a beautiful sunset")
    assert result['status'] == 'APPROVED'

def test_unsafe_prompt_refused():
    safety = SafetyManager()
    result = safety.moderate_prompt("violent content here")
    assert result['status'] == 'REFUSED'
```

**Integration Tests:**
```python
# test_pipeline.py
def test_end_to_end_generation():
    pipeline = LumaForgePipeline()
    result = pipeline.generate(
        prompt="test prompt",
        mock=True  # Fast mock mode
    )
    assert result['image'] is not None
    assert result['used_mock'] == True
```

**API Tests:**
```python
# test_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_generate_endpoint():
    response = client.post('/api/generate', json={
        'prompt': 'test',
        'mock': True
    })
    assert response.status_code == 200
    assert 'image_b64' in response.json()
```

### Performance Profiling

**Memory Profiling:**
```python
import tracemalloc

tracemalloc.start()

# Your generation code here
result = pipeline.generate(...)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

**Time Profiling:**
```python
import time

start = time.time()
result = pipeline.generate(...)
print(f"Total: {time.time() - start:.2f}s")
print(f"Model inference: {result['latency_sec']:.2f}s")
```

### Logging Best Practices

**Structured Logging:**
```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log generation events
logger.info(json.dumps({
    'event': 'generation_complete',
    'prompt_length': len(prompt),
    'latency_sec': result['latency_sec'],
    'memory_mb': result['memory_used_mb'],
    'seed': result['seed']
}))
```

## Deployment

### Production Considerations

**1. Model Caching:**
- Keep model loaded in memory
- Implement warmup requests
- Use model versioning

**2. Queue Management:**
- Implement proper task queue (Celery/Redis)
- Handle concurrent requests
- Set timeout policies

**3. Error Handling:**
- Graceful degradation
- Retry mechanisms
- User-friendly error messages

**4. Monitoring:**
- Track generation latency
- Monitor memory usage
- Log safety refusals
- Track API usage

**5. Security:**
- Rate limiting per IP
- API key authentication
- Input validation
- Output sanitization

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY model/ ./model/
COPY main.py .

# Download model
RUN python model/download_sd35.py

EXPOSE 8000

CMD ["uvicorn", "model.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  lumaforge-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./weights:/app/weights
    environment:
      - DEVICE=mps
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
  
  lumaforge-web:
    build: ./web
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - lumaforge-api

volumes:
  ollama-data:
```

### Scaling Strategies

**Horizontal Scaling:**
- Load balancer in front of API instances
- Shared model cache (Redis)
- Distributed task queue
- Centralized logging

**Vertical Scaling:**
- Larger GPU memory
- More CPU cores for Ollama
- SSD for faster model loading
- RAM for model caching

**Optimization:**
- Model quantization (8-bit)
- Batch inference where possible
- Async I/O for API
- CDN for static assets

## Roadmap & Future Features

### Short Term (Next Release)

**Enhanced Generation:**
- ControlNet integration for pose/depth control
- Inpainting with automatic mask detection
- Outpainting with context awareness
- Video generation (frame-by-frame)

**UI Improvements:**
- Generation history gallery
- Favorite/bookmark system
- Prompt library with templates
- Real-time preview during generation

**Performance:**
- Core ML optimization for Neural Engine
- Multi-model support with hot-swapping
- Quantized model variants
- Progressive image preview

### Medium Term (3-6 Months)

**Advanced Features:**
- Multi-image generation (variations)
- Style transfer with reference images
- Face swapping capabilities
- Text-to-3D preview
- Animation timeline editor

**Enterprise Features:**
- Multi-user support
- API key management
- Usage analytics dashboard
- Custom model training UI
- Team collaboration tools

**Platform Support:**
- iOS app (Core ML)
- iPad with Apple Pencil support
- macOS menu bar app
- CLI improvements

### Long Term (6-12 Months)

**AI Enhancements:**
- Custom diffusion models (1B-2B params)
- Improved safety classification
- Better prompt understanding
- Multi-modal generation
- Real-time editing

**Integration:**
- Figma plugin
- Photoshop extension
- Blender addon
- Unity/Unreal integration
- REST API marketplace

**Community:**
- Model sharing marketplace
- Prompt community library
- User-submitted LoRAs
- Competition/challenges
- Tutorial content

## Evaluation Metrics

### Quality Metrics

**Prompt Adherence:**
- Measured via CLIP score
- Human evaluation (1-5 scale)
- Object detection accuracy
- Text rendering accuracy

**Visual Quality:**
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Aesthetic predictor score
- Human preference rating

**Safety Metrics:**
- Precision: % of actual unsafe flagged correctly
- Recall: % of unsafe content caught
- False positive rate
- User report rate

### Performance Metrics

**Speed:**
- P50/P95/P99 latency percentiles
- Time to first pixel
- Full generation time
- Queue wait time

**Resource Usage:**
- Peak memory consumption
- Average memory usage
- GPU utilization
- CPU usage during idle

**Reliability:**
- Success rate (%)
- Error rate by type
- Timeout frequency
- Recovery time

### Business Metrics

**Engagement:**
- Daily active users
- Generations per user
- Session duration
- Return rate

**Quality:**
- User satisfaction score
- Net Promoter Score (NPS)
- Support ticket rate
- Feature usage distribution

## Contributing Guidelines

### Code Style

**Python:**
- Follow PEP 8
- Type hints for all functions
- Docstrings for public APIs
- Max line length: 100

**TypeScript:**
- ESLint configuration provided
- Prettier for formatting
- React hooks conventions
- Component organization

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Run linters and formatters
5. Update documentation
6. Submit PR with description
7. Address review feedback

### Commit Convention

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Testing
- `chore`: Maintenance

**Example:**
```
feat(pipeline): add ControlNet support

- Integrate ControlNet for pose control
- Add canny edge detection
- Update API endpoint
- Add UI controls

Closes #123
```

## License & Attribution

### Project License
LumaForge is released under [MIT License] - allowing commercial and personal use with attribution.

### Model Licenses

**Stable Diffusion 3.5 Medium:**
- License: Stability AI Community License
- Commercial use allowed with restrictions
- Attribution required

**Stable Diffusion 2.1:**
- License: CreativeML Open RAIL++-M
- Responsible AI usage required
- Commercial use allowed

### Third-Party Dependencies

**Core Libraries:**
- PyTorch (BSD License)
- Diffusers (Apache 2.0)
- Transformers (Apache 2.0)
- FastAPI (MIT License)
- Next.js (MIT License)

**Attribution:**
When using generated images commercially, consider:
- Disclosing AI generation
- Following model license terms
- Respecting safety guidelines
- Proper data handling

## Glossary

**CFG (Classifier-Free Guidance):** Parameter controlling how closely the model follows the prompt (guidance_scale)

**Diffusion Model:** Neural network that learns to denoise random noise into images

**Inference:** The process of generating an image from a trained model

**Latent Space:** Compressed representation where diffusion occurs

**LoRA:** Low-Rank Adaptation for efficient fine-tuning

**MPS:** Metal Performance Shaders, Apple's GPU acceleration

**Negative Prompt:** Terms to avoid in generation

**Sampler/Scheduler:** Algorithm controlling the denoising process

**Seed:** Random initialization value for reproducible generation

**Steps:** Number of denoising iterations (more = higher quality, slower)

**Strength:** How much img2img transformation to apply (0-1)

**VRAM/Unified Memory:** GPU memory used for model and tensors

## Support & Resources

### Documentation
- **Project README:** `/README.md`
- **API Docs:** `http://localhost:8000/docs` (when running)
- **PRD:** `/PRD.txt`
- **This Document:** `/PROJECT_DOCUMENTATION.md`

### Community
- **GitHub Issues:** Bug reports and feature requests
- **Discussions:** General questions and ideas
- **Discord:** Real-time chat and support (if available)

### Learning Resources
- **Stable Diffusion Papers:** Research foundations
- **Diffusers Documentation:** Library usage
- **Apple ML Docs:** MPS optimization guides
- **Ollama Docs:** LLM integration

### Getting Help

**For Bugs:**
1. Check existing issues
2. Provide reproduction steps
3. Include system information
4. Share error logs

**For Features:**
1. Search existing requests
2. Describe use case
3. Explain expected behavior
4. Discuss implementation ideas

**For Questions:**
1. Check documentation first
2. Search closed issues
3. Ask in discussions
4. Be specific and detailed

---

## Conclusion

LumaForge represents a comprehensive, production-ready AI image generation platform optimized for Apple Silicon. By combining cutting-edge diffusion models with intelligent safety systems, local LLM enhancement, and a modern web interface, it provides a complete solution for creative professionals, developers, and AI enthusiasts who value privacy, performance, and quality.

The modular architecture ensures maintainability and extensibility, while the focus on local-first processing guarantees user privacy and offline capability. With robust safety mechanisms, comprehensive API coverage, and active development, LumaForge is positioned as a leading open-source alternative to cloud-based image generation services.

**Project Status:** Active Development  
**Current Version:** 1.1  
**Last Updated:** 2026  
**Maintainers:** [Project Team]  

For the latest updates, visit the project repository and join the community discussions.

---

*This documentation was generated by analyzing the complete LumaForge codebase and architecture. For specific implementation details, refer to the source code and inline comments.*
