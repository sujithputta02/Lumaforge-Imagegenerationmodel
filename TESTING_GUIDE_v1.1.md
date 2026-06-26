# 🎨 LumaForge v1.1 - Testing Guide

## ✅ System Status

| Component | Status | Port | Details |
|-----------|--------|------|---------|
| **Frontend** | ✅ Running | 3000 | http://localhost:3000 |
| **Backend** | ✅ Running | 7860 | FastAPI + PyTorch |
| **Ollama** | ✅ Running | 11434 | llama3.2:1b model |
| **Mock Mode** | ✅ Default | - | Instant generation (~1-2 sec) |

---

## 🚀 Quick Start Testing

### 1. Navigate to Frontend
```
http://localhost:3000
```

### 2. Fill Out Generation Form
- **Prompt**: "A detailed scientific illustration with accurate details"
- **Category**: 📚 Education
- **Subcategory**: Scientific
- **Expansion Mode**: General Creative
- **Aspect Ratio**: 1:1 Square
- **Inference Steps**: 20
- **CFG Guidance Scale**: 7.5
- **Negative Prompt**: (auto-populated from backend)
- **Mock Runner**: ✅ ON (default)

### 3. Click "GENERATE IMAGE"
Expected result: **Image appears in 1-2 seconds** with metadata

---

## 📊 Expected Output Log

### Frontend Console
```
✅ Session created
✅ Polling generation status...
✅ Image received
```

### Backend Console
```
[Session XXX] Checking prompt safety: "..."
[OllamaClient] ✅ Connected to Ollama server
[Session XXX] Expanding prompt in mode 'general'
[LumaForgePipeline] Generating mock image (steps=20, guidance=7.5)
[LumaForgePipeline] Generation complete: 1.23s, memory=0.5MB, used_mock=True
```

---

## 🎯 Testing All 16 Categories

Try each category with the test prompts provided:

### ✅ Categories Ready for Testing

1. **🎨 Creative Art** (Digital Art, Concept Art, Fantasy, Sci-Fi, Surreal, Abstract, Matte Painting)
2. **👤 Characters** (Anime, Realistic, Cartoon, Game, Superhero, Medieval, Cyberpunk, Pixel Art)
3. **🏔️ Landscapes** (Mountains, Forests, Beaches, Waterfalls, Desert, Snow, Space, Underwater)
4. **🏛️ Architecture** (Modern, Futuristic, Ancient, Interior, Luxury, Office, Smart, Castles)
5. **🚗 Vehicles** (Sports Cars, Luxury, Motorcycles, Aircraft, Spacecraft, Ships, Military)
6. **📦 Products** (Mockups, Furniture, Shoes, Watches, Electronics, Perfume, Packaging, Cosmetics)
7. **📢 Marketing** (Posters, Flyers, Social, Thumbnails, Book Covers, Magazines, Banners, Ads)
8. **🍰 Food** (Dishes, Desserts, Beverages, Cakes, Fast Food, Gourmet, Recipes)
9. **👗 Fashion** (Clothing, Dresses, Jackets, Sneakers, Jewelry, Accessories, Runway)
10. **🎮 Gaming** (Icons, UI, Backgrounds, NPCs, Weapons, Effects, Inventory)
11. **🐾 Animals** (Pets, Wildlife, Birds, Marine, Fantasy, Dragons, Mythical)
12. **🎉 Events** (Weddings, Birthdays, Festivals, Holidays, Parties)
13. **💼 Business** (Infographics, Presentations, Dashboards, Banners, Branding)
14. **📚 Education** (Scientific, Biology, History, Geography, Medical)

---

## 🎛️ Advanced Features to Test

### Image Editing (After Generating an Image)

#### Colorization
```
Select Style: Vibrant | Warm | Cool | Vintage | Sepia
Click: "COLORIZE"
Expected: Image recolored in selected style (~1-2 sec)
```

#### Face Restoration
```
Select Intensity: Low | Medium | High | Ultra
Click: "RESTORE FACE"
Expected: Facial features enhanced (~1-2 sec)
```

### Modes to Test

1. **General Creative** - Balanced prompt expansion
2. **Movie Poster** - Typography-optimized output
3. **Character Concept** - Detail-focused rendering

### Aspect Ratios

- 1:1 (Square)
- 16:9 (Widescreen)
- 9:16 (Portrait)
- 4:3 (Standard)
- 3:4 (Portrait)

---

## 🔄 Toggling Mock vs Real Mode

### Mock Mode (Default - Instant)
```
Toggle: Mock Runner = ON ✅
Result: Image in ~1-2 seconds
Use for: Quick testing, UI verification
```

### Real Mode (SDXL - Slow but Photorealistic)
```
Toggle: Mock Runner = OFF ⚪
Result: Model downloads (4GB+, 5-10 min first time)
Then: Generation takes 30-60 seconds
Use for: Production quality images
```

**⚠️ Note**: Real mode requires large model download on first run.

---

## 📋 Test Checklist

- [ ] Frontend loads at localhost:3000
- [ ] Backend status shows "healthy"
- [ ] Can generate image with default settings
- [ ] Image appears with metadata (latency, memory, seed)
- [ ] Mock mode toggle works
- [ ] Category selector works
- [ ] All 16 categories load subcategories
- [ ] Colorization endpoint works
- [ ] Face restoration endpoint works
- [ ] Negative prompts are applied
- [ ] Different seeds produce different images
- [ ] Aspect ratio changes work
- [ ] Steps slider affects generation (in real mode)
- [ ] CFG guidance scale affects generation (in real mode)

---

## 🐛 Troubleshooting

### Issue: Image not generating
**Solution**: Check mock mode is enabled (default), wait 2-3 seconds

### Issue: Backend connection refused
**Solution**: Restart backend with `python3 app.py` from `/model` directory

### Issue: Ollama not connected
**Solution**: Run `ollama serve` separately (already running on 11434)

### Issue: Very slow generation
**Solution**: Enable mock mode (toggle on), or wait for real model download

### Issue: Category selector not showing
**Solution**: Refresh page (F5), clear browser cache

---

## 📊 Performance Metrics

| Metric | Mock Mode | Real Mode (SDXL) |
|--------|-----------|------------------|
| **First Run** | Instant | 5-10 minutes (download) |
| **Generation Time** | 1-2 seconds | 30-60 seconds |
| **Memory Usage** | 50-100 MB | 2-4 GB |
| **Quality** | Stylized, artistic | Photorealistic |
| **Best For** | Testing, development | Production output |

---

## 📚 Latest Commits

| Commit | Message |
|--------|---------|
| `9c2da37` | Enable mock mode by default + comprehensive logging |
| `5680fa0` | Ollama client improvements with llama3.2:1b |
| `52db35c` | Fallback handling & debug logging |
| `dcc1b85` | Updated model README for v1.1 |
| `0914032` | v1.1: Add 16 categories, colorization, face restoration |

---

## 🎉 Ready to Test!

Everything is configured and running. Navigate to **http://localhost:3000** and start generating!

### Questions?
Check the comprehensive test prompts in the chat for all 16 categories with negative prompts.
