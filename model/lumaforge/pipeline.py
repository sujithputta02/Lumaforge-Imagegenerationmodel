import os
import time
import random
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
from PIL.PngImagePlugin import PngInfo
import numpy as np

class LumaForgePipeline:
    def __init__(self, model_id="stabilityai/stable-diffusion-3.5-medium", device="mps", ollama_client=None):
        self.model_id = model_id
        self.device = device if torch.backends.mps.is_available() and device == "mps" else "cpu"
        self.pipe = None
        self.is_loaded = False
        self.ollama_client = ollama_client
        print(f"[LumaForgePipeline] Initialized SD 3.5 Medium pipeline with device: {self.device}")

    def load_model(self):
        """Loads SD 3.5 Medium pipeline - latest Stability AI model."""
        if self.is_loaded:
            return True
            
        print(f"[LumaForgePipeline] Loading SD 3.5 Medium model onto {self.device}...")
        print(f"[LumaForgePipeline] Checking local cache at ~/.cache/huggingface/...")
        try:
            from diffusers import StableDiffusion3Pipeline
            import os
            
            # Use fp16 for MPS
            torch_dtype = torch.float16
            
            # Set cache directory explicitly
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            
            print(f"[LumaForgePipeline] Loading SD 3.5 Medium (this will download ~5-6GB on first run)...")
            
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                local_files_only=False
            )
                
            print(f"[LumaForgePipeline] ✅ SD 3.5 Medium loaded successfully")
            # Memory optimization & Device placement
            if self.device in ["mps", "cuda"]:
                try:
                    print(f"[LumaForgePipeline] Enabling model CPU offloading for {self.device} memory optimization...")
                    self.pipe.enable_model_cpu_offload(device=self.device)
                    print(f"[LumaForgePipeline] ✅ Model CPU offloading enabled.")
                except Exception as e:
                    print(f"[LumaForgePipeline Warning] Failed to enable CPU offloading: {e}. Falling back to full device load.")
                    print(f"[LumaForgePipeline] Moving pipeline to {self.device}...")
                    self.pipe.to(self.device)
                    print(f"[LumaForgePipeline] ✅ Pipeline successfully moved to {self.device}")
                    if self.device == "mps":
                        print(f"[LumaForgePipeline] Enabling attention slicing for MPS memory optimization...")
                        self.pipe.enable_attention_slicing()
                        print(f"[LumaForgePipeline] ✅ Attention slicing enabled.")
            else:
                print(f"[LumaForgePipeline] Moving pipeline to {self.device}...")
                self.pipe.to(self.device)
                print(f"[LumaForgePipeline] ✅ Pipeline successfully moved to {self.device}")
                
            self.is_loaded = True
            print("[LumaForgePipeline] ✅ SD 3.5 Medium ready for inference!")
            return True
        except Exception as e:
            print(f"[LumaForgePipeline Error] Failed to load SD 3.5 Medium: {e}")
            print(f"[LumaForgePipeline] Model needs to be downloaded first.")
            self.is_loaded = False
            return False

    def generate(self, prompt: str, aspect_ratio="1:1", steps=20, seed=None, guidance_scale=7.5, negative_prompt="", mock=False) -> dict:
        """
        Generates an image from a prompt.
        If mock=True or model loading fails, runs in Mock Mode to generate a high-quality stylized abstract visual.
        """
        start_time = time.time()
        
        # Determine image dimensions based on aspect ratio
        width, height = self._get_dimensions(aspect_ratio)
        
        # Set random seed if not provided
        if seed is None or seed == -1:
            seed = random.randint(0, 9999999)
            
        # Get starting memory
        start_mem_bytes = self._get_mps_memory()
        
        image = None
        used_mock = False
        gen_prompt = prompt
        
        # Extract quoted titles for negative prompt and overlay logic
        import re
        titles = re.findall(r'"([^"]+)"', prompt)
        if not titles:
            titles = re.findall(r"'([^']+)'", prompt)
        
        if mock:
            print(f"[LumaForgePipeline] Generating mock image (steps={steps}, guidance={guidance_scale})")
            image = self._generate_mock_image(prompt, width, height, aspect_ratio, seed)
            used_mock = True
            # Simulate processing time
            time.sleep(1.5)
        else:
            # SD 3.5 Medium: Use Ollama to optimize prompt for 77-token limit
            prompt_lower = prompt.lower()
            
            # Use Ollama to intelligently compress the prompt if needed
            if self.ollama_client:
                print(f"[LumaForgePipeline] Optimizing prompt for SD 3.5 Medium token limit...")
                optimization = self.ollama_client.optimize_prompt_for_sd35(prompt, max_tokens=256)
                
                if optimization["was_compressed"]:
                    print(f"[LumaForgePipeline] ✅ Prompt optimized: {optimization['original_tokens']} → {optimization['final_tokens']} tokens")
                    prompt = optimization["optimized_prompt"]
                else:
                    print(f"[LumaForgePipeline] ✅ Prompt already optimal ({optimization['original_tokens']} tokens)")
            else:
                print(f"[LumaForgePipeline] ⚠️  Ollama not available, using original prompt")
            
            # OPTIMIZED NEGATIVE PROMPT (essential negatives only for SD 3.5 Medium)
            core_negatives = "low quality, blurry"
            
            # Add facial negatives for character/portrait images
            if any(kw in prompt_lower for kw in ["face", "portrait", "character", "person", "wizard", "man", "woman"]):
                core_negatives = f"{core_negatives}, bad anatomy"
            
            # Style-aware exclusions (minimal)
            if "photorealistic" in prompt_lower or "photo" in prompt_lower:
                core_negatives = f"{core_negatives}, cartoon"
            elif "anime" in prompt_lower:
                core_negatives = f"{core_negatives}, photorealistic"
            
            if not negative_prompt:
                negative_prompt = core_negatives
            else:
                negative_prompt = f"{negative_prompt}, {core_negatives}"

            # If titles found, suppress text generation
            if titles:
                negative_prompt = f"{negative_prompt}, text, letters"
            
            # Token estimation (rough: ~1.3 chars per token)
            prompt_tokens = len(prompt) // 1.3
            neg_tokens = len(negative_prompt) // 1.3
            
            print(f"[LumaForgePipeline] Token estimate: prompt ~{int(prompt_tokens)}, negative ~{int(neg_tokens)}")
            if prompt_tokens > 256:
                print(f"[LumaForgePipeline] ⚠️  Prompt may be truncated (exceeds 256 tokens)")
                    
            loaded = self.load_model()
            if not loaded:
                print("[LumaForgePipeline] Falling back to Mock Generation due to loading failure.")
                image = self._generate_mock_image(prompt, width, height, aspect_ratio, seed)
                used_mock = True
                time.sleep(1.5)
            else:
                try:
                    # 8. SD 3.5 OPTIMAL PARAMETERS
                    optimized_steps = 28
                    optimized_guidance = 4.5
                    
                    print(f"[LumaForgePipeline] SD 3.5 Medium inference: steps={optimized_steps}, guidance={optimized_guidance}, seed={seed}")
                    print(f"[LumaForgePipeline] Prompt: {prompt[:100]}...")
                    print(f"[LumaForgePipeline] Negative: {negative_prompt[:80]}...")
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    
                    # Run SD 3.5 Medium diffusion
                    output = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=optimized_steps,
                        guidance_scale=optimized_guidance,
                        width=width,
                        height=height,
                        generator=generator
                    )
                    image = output.images[0]
                    
                    print(f"[LumaForgePipeline] ✅ SD 3.5 Medium inference completed")
                except Exception as e:
                    print(f"[LumaForgePipeline Error] Inference failed: {e}. Falling back to mock image.")
                    image = self._generate_mock_image(prompt, width, height, aspect_ratio, seed)
                    used_mock = True
                    
        # Apply programmatic typography overlay for actual poster generations
        if not used_mock and "poster" in prompt.lower() and titles:
            title = titles[0]
            print(f"[LumaForgePipeline] Applying programmatic typography overlay for title: '{title}'")
            image = self._overlay_poster_typography(image, title)
            
        latency_sec = time.time() - start_time
        end_mem_bytes = self._get_mps_memory()
        
        # Calculate memory footprint delta or absolute usage
        memory_used_mb = max(0.0, (end_mem_bytes - start_mem_bytes) / (1024 * 1024))
        if memory_used_mb == 0.0 and self.device == "mps":
            # Show current absolute allocation if delta is 0
            memory_used_mb = end_mem_bytes / (1024 * 1024)
            
        # Apply LumaForge low-transparency watermark logo overlay
        image = self._overlay_lumaforge_logo(image)
        
        print(f"[LumaForgePipeline] Generation complete: {latency_sec:.2f}s, memory={memory_used_mb:.1f}MB, used_mock={used_mock}")
        
        # Construct PNG Metadata
        metadata = PngInfo()
        metadata.add_text("prompt", str(gen_prompt))
        metadata.add_text("negative_prompt", str(negative_prompt))
        metadata.add_text("seed", str(seed))
        metadata.add_text("steps", str(steps))
        metadata.add_text("guidance_scale", str(guidance_scale))
        metadata.add_text("model_id", str(self.model_id))
        metadata.add_text("software", "LumaForge AuraGen Core")
        
        return {
            "image": image,
            "pnginfo": metadata,
            "latency_sec": latency_sec,
            "memory_used_mb": memory_used_mb,
            "seed": seed,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "used_mock": used_mock,
            "device": self.device
        }

    def generate_img2img(self, image: Image.Image, prompt: str, strength=0.5, steps=20, seed=None, guidance_scale=7.5, negative_prompt="", mock=False) -> dict:
        """
        Generates a new image based on an input image and a prompt.
        If mock=True or model loading fails, runs in Mock Mode to blend the input with a retro-wave composition.
        """
        start_time = time.time()
        
        # Set random seed if not provided
        if seed is None or seed == -1:
            seed = random.randint(0, 9999999)
            
        # Get starting memory
        start_mem_bytes = self._get_mps_memory()
        
        used_mock = False
        output_image = None
        is_cartoon = False
        
        # Extract quoted titles for negative prompt and overlay logic
        import re
        titles = re.findall(r'"([^"]+)"', prompt)
        if not titles:
            titles = re.findall(r"'([^']+)'", prompt)
            
        # Standardize input image dimensions to match standard generation size (e.g. 512x512)
        width, height = 512, 512
        input_resized = image.convert("RGB").resize((width, height))
        
        if mock:
            output_image = self._generate_mock_img2img(input_resized, prompt, strength, seed)
            used_mock = True
            time.sleep(1.5)
        else:
            p_lower = prompt.lower()
            is_cartoon = any(keyword in p_lower for keyword in ["cartoon", "anime", "ghibli", "sketch", "drawing", "illustration"])
            
            if is_cartoon:
                # Cap strength to preserve exact facial structure and prevent morphing
                strength = min(strength, 0.32)
                
                # Append high-fidelity style descriptions to prompt
                if "ghibli" in p_lower:
                    prompt = f"{prompt}, studio ghibli style hand-drawn animation, soft lighting, warm aesthetic, detailed scenery, anime key visual, masterpiece"
                elif "anime" in p_lower or "cartoon" in p_lower:
                    prompt = f"{prompt}, professional anime key visual, clean lineart, cell shaded colors, vibrant lighting, highly detailed illustration, masterpiece"
                elif "sketch" in p_lower or "drawing" in p_lower:
                    prompt = f"{prompt}, highly detailed pencil sketch art, hand-drawn pencil shading, clean white paper background, high contrast lines"
                
                # Append specialized style-preserving negative prompts to avoid melting/morphing
                cartoon_neg = "photorealistic, photo, 3d render, morphed faces, deformed eyes, extra limbs, bad anatomy, blurry, low resolution, low quality"
                if not negative_prompt:
                    negative_prompt = cartoon_neg
                else:
                    negative_prompt = f"{negative_prompt}, {cartoon_neg}"
            else:
                # Quality enhancement trigger words for normal images
                if "high quality" not in p_lower and "high-resolution" not in p_lower:
                    prompt = f"{prompt}, high-resolution, 8k, detailed, sharp focus"
                    
                # Quality enhancement negative prompt filter
                quality_neg = "blurry, blur, out of focus, low quality, low resolution, duplicate, bad anatomy, deformed, distorted"
                if not negative_prompt:
                    negative_prompt = quality_neg
                else:
                    negative_prompt = f"{negative_prompt}, {quality_neg}"

            # If a title is found in the prompt, suppress model text generation to avoid double/garbled lettering
            if titles:
                neg_text = "text, letters, words, writing, signage, gibberish lettering, garbled text"
                negative_prompt = f"{negative_prompt}, {neg_text}"
                    
            loaded = self.load_model()
            if not loaded:
                print("[LumaForgePipeline] Falling back to Mock Img2Img due to loading failure.")
                output_image = self._generate_mock_img2img(input_resized, prompt, strength, seed)
                used_mock = True
                time.sleep(1.5)
            else:
                try:
                    from diffusers import StableDiffusionImg2ImgPipeline
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    
                    # Sharing pipeline components to save memory
                    if hasattr(StableDiffusionImg2ImgPipeline, "from_pipe"):
                        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(self.pipe)
                    else:
                        img2img_pipe = StableDiffusionImg2ImgPipeline(
                            vae=self.pipe.vae,
                            text_encoder=self.pipe.text_encoder,
                            tokenizer=self.pipe.tokenizer,
                            unet=self.pipe.unet,
                            scheduler=self.pipe.scheduler,
                            safety_checker=self.pipe.safety_checker,
                            feature_extractor=self.pipe.feature_extractor
                        )
                    
                    # Run img2img diffusion
                    output = img2img_pipe(
                        prompt=prompt,
                        image=input_resized,
                        strength=strength,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
                    output_image = output.images[0]
                except Exception as e:
                    print(f"[LumaForgePipeline Error] Img2Img inference failed: {e}. Falling back to mock blend.")
                    output_image = self._generate_mock_img2img(input_resized, prompt, strength, seed)
                    used_mock = True
                    
        # Apply programmatic typography overlay for poster modes
        if not used_mock and "poster" in prompt.lower() and titles:
            title = titles[0]
            output_image = self._overlay_poster_typography(output_image, title)
            
        # For cartoon/anime styles in real mode, apply pixel-accurate adaptive detail reinforcement and structural blend
        if not used_mock and is_cartoon and output_image is not None:
            try:
                import numpy as np
                # Convert original and generated images to float arrays
                orig_arr = np.array(input_resized, dtype=float)
                gen_arr = np.array(output_image, dtype=float)
                
                # 1. High-frequency details transfer (high-pass filter on original grayscale channel)
                orig_gray = ImageOps.grayscale(input_resized)
                orig_gray_blurred = orig_gray.filter(ImageFilter.GaussianBlur(radius=1.5))
                orig_gray_arr = np.array(orig_gray_blurred, dtype=float)
                orig_y_arr = np.array(orig_gray, dtype=float)
                
                # High-pass values represent sharp face structures and suit web lines
                high_pass = orig_y_arr - orig_gray_arr
                high_pass_3d = np.expand_dims(high_pass, axis=2) # Broadcast to 3 channels
                
                # Add high-frequency original details to generated output (blending factor 0.30)
                gen_enhanced_arr = gen_arr + 0.30 * high_pass_3d
                
                # 2. Radial Face Protection Mask (centered at face region)
                # This keeps the face region extremely accurate to the original photo while
                # allowing the background and shoulders to take on the full cartoon stylization.
                y_coords, x_coords = np.ogrid[:height, :width]
                center_y, center_x = int(height * 0.44), int(width * 0.5) # centered at face
                distance_sq = (y_coords - center_y)**2 + (x_coords - center_x)**2
                
                radius = 110.0
                face_mask = np.exp(-distance_sq / (2.0 * (radius**2)))
                
                # Face blend: 55% original, 45% generated. Background: 10% original, 90% generated.
                blend_factor = 0.10 + 0.45 * face_mask
                blend_factor_3d = np.expand_dims(blend_factor, axis=2)
                
                # Pixel-by-pixel composite
                composited_arr = orig_arr * blend_factor_3d + gen_enhanced_arr * (1.0 - blend_factor_3d)
                output_image = Image.fromarray(np.clip(composited_arr, 0, 255).astype(np.uint8))
                
                # 3. Dreamy Ghibli Bloom Glow (soft blurred highlight overlay)
                glow = output_image.filter(ImageFilter.GaussianBlur(6))
                output_image = Image.blend(output_image, glow, 0.12)
                
                # 4. Stylized color contrast and saturation boost
                color_enhancer = ImageEnhance.Color(output_image)
                vibrant = color_enhancer.enhance(1.25)
                contrast_enhancer = ImageEnhance.Contrast(vibrant)
                output_image = contrast_enhancer.enhance(1.06)
                
                print("[LumaForgePipeline] Successfully executed radial face protection post-processing.")
            except Exception as e:
                print(f"[LumaForgePipeline Warning] Adaptive detail restoration failed: {e}")
            
        latency_sec = time.time() - start_time
        end_mem_bytes = self._get_mps_memory()
        
        memory_used_mb = max(0.0, (end_mem_bytes - start_mem_bytes) / (1024 * 1024))
        if memory_used_mb == 0.0 and self.device == "mps":
            memory_used_mb = end_mem_bytes / (1024 * 1024)
            
        # Apply logo watermark
        output_image = self._overlay_lumaforge_logo(output_image)
        
        # Construct PNG Metadata
        metadata = PngInfo()
        metadata.add_text("prompt", str(prompt))
        metadata.add_text("negative_prompt", str(negative_prompt))
        metadata.add_text("seed", str(seed))
        metadata.add_text("steps", str(steps))
        metadata.add_text("guidance_scale", str(guidance_scale))
        metadata.add_text("strength", str(strength))
        metadata.add_text("model_id", str(self.model_id))
        metadata.add_text("software", "LumaForge AuraGen Core")

        return {
            "image": output_image,
            "pnginfo": metadata,
            "latency_sec": latency_sec,
            "memory_used_mb": memory_used_mb,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "used_mock": used_mock,
            "device": self.device
        }

    def _generate_mock_img2img(self, image: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
        """
        Generates a task-aware mock image editing output.
        Inspects prompt keywords to perform stylized PIL-based transformations (Ghibli paint, snow overlay,
        color shift, sketch contours, or background replacement) blended according to strength.
        """
        import numpy as np
        p = prompt.lower()
        width, height = image.size
        edited = image.copy()
        
        # 1. Style Transfer (Ghibli / anime / painting / watercolor / sketch / cartoon)
        if any(keyword in p for keyword in ["ghibli", "anime", "painting", "watercolor", "sketch", "cartoon"]):
            if "sketch" in p:
                # Pencil sketch effect using highly optimized vectorized NumPy math
                gray = ImageOps.grayscale(edited)
                inverted = ImageOps.invert(gray)
                blurred = inverted.filter(ImageFilter.GaussianBlur(8))
                
                # NumPy vectorized dodge blend
                gray_arr = np.array(gray, dtype=float)
                blurred_arr = np.array(blurred, dtype=float)
                
                denominator = 255.0 - blurred_arr
                denominator[denominator == 0] = 1e-5
                
                dodge_arr = (gray_arr * 255.0) / denominator
                dodge_arr = np.clip(dodge_arr, 0, 255).astype(np.uint8)
                edited = Image.fromarray(dodge_arr).convert("RGB")
                
                # Boost sketch contrast to make outlines pop
                contrast = ImageEnhance.Contrast(edited)
                edited = contrast.enhance(1.7)
            else:
                # High-fidelity cell-shaded cartoon/anime/Ghibli style
                img_arr = np.array(edited, dtype=float)
                
                # A: Bilateral Filter (edge-preserving texture smoothing)
                def fast_bilateral_filter(arr, sigma_s=3.0, sigma_r=25.0):
                    h_val, w_val, c_val = arr.shape
                    out = np.zeros_like(arr)
                    w_sum = np.zeros((h_val, w_val, 1))
                    
                    # 5x5 window
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if dx == 0 and dy == 0:
                                spatial_w = 1.0
                            else:
                                spatial_w = np.exp(-(dx**2 + dy**2) / (2.0 * (sigma_s**2)))
                                
                            neighbor = np.roll(arr, shift=(dy, dx), axis=(0, 1))
                            diff = arr - neighbor
                            color_dist_sq = np.sum(diff**2, axis=2, keepdims=True)
                            color_w = np.exp(-color_dist_sq / (2.0 * (sigma_r**2)))
                            
                            total_w = spatial_w * color_w
                            out += neighbor * total_w
                            w_sum += total_w
                            
                    return out / (w_sum + 1e-5)
                
                smoothed_arr = fast_bilateral_filter(img_arr, sigma_s=2.5, sigma_r=20.0)
                smoothed = Image.fromarray(np.clip(smoothed_arr, 0, 255).astype(np.uint8))
                
                # B: YCbCr Luminance-only cell-shading (preserves skin tones and hues)
                ycbcr = smoothed.convert("YCbCr")
                y, cb, cr = ycbcr.split()
                
                y_blurred = y.filter(ImageFilter.GaussianBlur(1))
                y_arr = np.array(y_blurred, dtype=float)
                
                # Quantize Y channel to 5 beautiful stepped levels
                y_quant = np.zeros_like(y_arr)
                y_quant[y_arr < 55] = 40
                y_quant[(y_arr >= 55) & (y_arr < 110)] = 95
                y_quant[(y_arr >= 110) & (y_arr < 165)] = 150
                y_quant[(y_arr >= 165) & (y_arr < 220)] = 205
                y_quant[y_arr >= 220] = 255
                
                # Smooth transition borders
                y_new = Image.fromarray(y_quant.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
                shaded = Image.merge("YCbCr", (y_new, cb, cr)).convert("RGB")
                
                # C: Grayscale gradient-magnitude organic outline extraction (ink lines)
                gray = ImageOps.grayscale(smoothed)
                gray_blurred = gray.filter(ImageFilter.GaussianBlur(1.0))
                gray_arr = np.array(gray_blurred, dtype=float)
                
                grad_x = np.gradient(gray_arr, axis=1)
                grad_y = np.gradient(gray_arr, axis=0)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                max_grad = grad_mag.max()
                if max_grad > 0:
                    grad_mag = (grad_mag / max_grad) * 255.0
                
                # Soft threshold for anti-aliasing
                min_edge, max_edge = 15.0, 35.0
                edge_mask = np.zeros_like(grad_mag, dtype=float)
                mask_range = (grad_mag > min_edge) & (grad_mag < max_edge)
                edge_mask[mask_range] = (grad_mag[mask_range] - min_edge) / (max_edge - min_edge)
                edge_mask[grad_mag >= max_edge] = 1.0
                
                # Soften mask to organic ink stroke weight
                edge_mask_img = Image.fromarray((edge_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
                edge_mask_final = np.array(edge_mask_img, dtype=float) / 255.0
                
                shaded_arr = np.array(shaded, dtype=float)
                ink_color = np.array([32, 28, 38]) # Dark charcoal ink
                edge_mask_expanded = np.expand_dims(edge_mask_final, axis=2)
                
                final_arr = shaded_arr * (1.0 - edge_mask_expanded) + ink_color * edge_mask_expanded
                final_cartoon = Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8))
                
                # D: Volumetric Bloom / Highlight Glow
                glow_glow = final_cartoon.filter(ImageFilter.GaussianBlur(12))
                final_cartoon = Image.blend(final_cartoon, glow_glow, 0.18)
                
                # E: Saturation and Contrast boost
                color_enhancer = ImageEnhance.Color(final_cartoon)
                vibrant = color_enhancer.enhance(1.65)
                contrast_enhancer = ImageEnhance.Contrast(vibrant)
                edited = contrast_enhancer.enhance(1.15)
                
                # F: Ghibli Warm Temp Shift
                if "ghibli" in p:
                    r, g, b = edited.split()
                    r = r.point(lambda x: min(255, int(x * 1.05)))
                    b = b.point(lambda x: int(x * 0.93))
                    edited = Image.merge("RGB", (r, g, b))
                
        # 2. Lighting & Weather Changes (snow / winter / rain / night)
        if "snow" in p or "winter" in p or "rain" in p or "night" in p:
            if "snow" in p or "winter" in p:
                r, g, b = edited.split()
                r = r.point(lambda x: int(x * 0.90))
                b = b.point(lambda x: min(255, int(x * 1.10)))
                edited = Image.merge("RGB", (r, g, b))
            elif "rain" in p or "storm" in p:
                brightness = ImageEnhance.Brightness(edited)
                edited = brightness.enhance(0.75)
                r, g, b = edited.split()
                r = r.point(lambda x: int(x * 0.92))
                b = b.point(lambda x: min(255, int(x * 1.06)))
                edited = Image.merge("RGB", (r, g, b))
            elif "night" in p:
                brightness = ImageEnhance.Brightness(edited)
                edited = brightness.enhance(0.60)
                r, g, b = edited.split()
                r = r.point(lambda x: int(x * 0.85))
                g = g.point(lambda x: int(x * 0.85))
                b = b.point(lambda x: min(255, int(x * 1.15)))
                edited = Image.merge("RGB", (r, g, b))

            draw = ImageDraw.Draw(edited, "RGBA")
            import random
            random.seed(seed)
            if "snow" in p or "winter" in p:
                snow_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                snow_draw = ImageDraw.Draw(snow_overlay)
                for _ in range(90):
                    rx = random.randint(0, width)
                    ry = random.randint(0, height)
                    r_size = random.choice([2, 3, 4, 5])
                    alpha = random.randint(120, 230)
                    snow_draw.ellipse([rx - r_size, ry - r_size, rx + r_size, ry + r_size], fill=(255, 255, 255, alpha))
                snow_overlay = snow_overlay.filter(ImageFilter.GaussianBlur(1.0))
                edited = Image.alpha_composite(edited.convert("RGBA"), snow_overlay).convert("RGB")
            elif "rain" in p:
                rain_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                rain_draw = ImageDraw.Draw(rain_overlay)
                for _ in range(150):
                    rx = random.randint(0, width)
                    ry = random.randint(0, height)
                    length = random.randint(10, 22)
                    width_val = random.choice([1, 2])
                    alpha = random.randint(80, 160)
                    rain_draw.line([(rx, ry), (rx - 3, ry + length)], fill=(210, 230, 255, alpha), width=width_val)
                rain_overlay = rain_overlay.filter(ImageFilter.GaussianBlur(0.8))
                edited = Image.alpha_composite(edited.convert("RGBA"), rain_overlay).convert("RGB")
                
        # 3. Object Addition (drone, spaceship, hat, car, etc.)
        if any(keyword in p for keyword in ["add a", "add some", "insert", "place"]):
            draw = ImageDraw.Draw(edited, "RGBA")
            cx, cy = int(width * 0.5), int(height * 0.4)
            draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=(0, 242, 254, 180), width=2)
            draw.line([(cx - 45, cy), (cx + 45, cy)], fill=(0, 242, 254, 150), width=1)
            draw.line([(cx, cy - 45), (cx, cy + 45)], fill=(0, 242, 254, 150), width=1)
            
            try:
                font_path = "/System/Library/Fonts/Helvetica.ttc"
                if not os.path.exists(font_path):
                    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
                font = ImageFont.truetype(font_path, 10)
            except Exception:
                font = ImageFont.load_default()
                
            draw.text((cx - 15, cy - 6), "ADD+", fill=(255, 255, 255, 240), font=font)

        # 4. Background Replacement (preserving people and objects perfectly!)
        if "background" in p or "replace the" in p:
            # Generate new background
            mock_bg = self._generate_mock_image(prompt, width, height, "1:1", seed)
            # Cut out foreground from input image
            foreground = self.remove_background(image, mock=True)
            # Paste foreground onto mock background using transparent mask channel
            mock_bg.paste(foreground, (0, 0), foreground)
            edited = mock_bg
            
        # 5. Color Modification
        if "color" in p or "recolor" in p or "blue" in p or "red" in p or "green" in p:
            tint_color = (0, 0, 255)
            if "red" in p:
                tint_color = (255, 0, 0)
            elif "green" in p:
                tint_color = (0, 255, 0)
            
            tint_layer = Image.new("RGB", (width, height), tint_color)
            edited = Image.blend(edited, tint_layer, 0.15)
            
        # Blend the modified image with the original image according to strength
        return Image.blend(image.convert("RGB"), edited.convert("RGB"), strength)

    def upscale(self, image: Image.Image, scale_factor: float = 2.0, mock: bool = False) -> dict:
        """
        Upscales the PIL image using high-quality LANCZOS interpolation
        and applies an unsharp mask to refine edges and details.
        """
        start_time = time.time()
        start_mem_bytes = self._get_mps_memory()
        
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # High fidelity resize
        resampled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Edge sharpening filter
        sharpened = resampled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Apply logo watermark on the final upscaled image to preserve quality
        final_image = self._overlay_lumaforge_logo(sharpened)
        
        latency_sec = time.time() - start_time
        end_mem_bytes = self._get_mps_memory()
        memory_used_mb = max(0.0, (end_mem_bytes - start_mem_bytes) / (1024 * 1024))
        
        return {
            "image": final_image,
            "latency_sec": latency_sec,
            "memory_used_mb": memory_used_mb,
            "width": new_width,
            "height": new_height,
            "device": self.device
        }

    def remove_background(self, image: Image.Image, mock: bool = False) -> Image.Image:
        """
        Removes the background of the image.
        If rembg is installed, uses it. Otherwise, runs a high-fidelity chroma-key/color threshold
        or foreground detection algorithm to create a transparent PNG.
        """
        try:
            if not mock:
                import rembg
                return rembg.remove(image)
        except ImportError:
            print("[LumaForgePipeline] rembg not found, falling back to PIL-based color-threshold background removal.")
        
        import numpy as np
        img = image.convert("RGBA")
        width, height = img.size
        
        # Sample corner pixels to find background color
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((width - 1, 0)),
            img.getpixel((0, height - 1)),
            img.getpixel((width - 1, height - 1))
        ]
        
        from collections import Counter
        bg_color = Counter(corners).most_common(1)[0][0]
        bg_r, bg_g, bg_b, _ = bg_color
        
        # Convert to numpy array
        img_arr = np.array(img)
        rgb = img_arr[:, :, :3].astype(float)
        alpha = img_arr[:, :, 3].copy()
        
        # Distance calculation in numpy
        bg_rgb = np.array([bg_r, bg_g, bg_b], dtype=float)
        dists = np.sqrt(np.sum((rgb - bg_rgb) ** 2, axis=2))
        
        threshold = 35.0
        # Smooth transition feathering:
        # below min_thresh: alpha is 0
        # above max_thresh: alpha is original alpha
        # in between: smooth interpolation
        min_thresh = max(0.0, threshold - 15.0)
        max_thresh = threshold + 15.0
        
        feathered_alpha = np.zeros_like(dists, dtype=float)
        # Keep foreground intact
        feathered_alpha[dists >= max_thresh] = alpha[dists >= max_thresh]
        # Interpolate transition margins
        mask = (dists > min_thresh) & (dists < max_thresh)
        ratio = (dists[mask] - min_thresh) / (max_thresh - min_thresh)
        feathered_alpha[mask] = alpha[mask] * ratio
        
        # Update alpha channel
        img_arr[:, :, 3] = np.clip(feathered_alpha, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        
        # Feather the edges of the alpha channel using a tiny Gaussian blur for a smooth professional cut
        r, g, b, alpha_channel = img.split()
        alpha_blurred = alpha_channel.filter(ImageFilter.GaussianBlur(1.0))
        return Image.merge("RGBA", (r, g, b, alpha_blurred))

    def _get_dimensions(self, aspect_ratio: str) -> tuple:
        """Returns standard dimensions based on aspect ratio."""
        mapping = {
            "1:1": (512, 512),
            "16:9": (768, 432),
            "9:16": (432, 768),
            "4:3": (640, 480),
            "3:4": (480, 640)
        }
        return mapping.get(aspect_ratio, (512, 512))

    def _get_mps_memory(self) -> int:
        """Returns the current allocated memory on Apple MPS in bytes."""
        if self.device == "mps":
            try:
                # Returns current memory allocated on MPS
                return torch.mps.current_allocated_memory()
            except Exception:
                return 0
        return 0

    def _restore_face(self, image: Image.Image) -> Image.Image:
        """
        Restores facial details and clarity using GFPGAN for crystal-clear faces.
        Falls back gracefully if GFPGAN not available.
        """
        try:
            from gfpgan import GFPGANer
            
            # Initialize GFPGAN
            restorer = GFPGANer(
                scale=2,
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=True,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            
            # Convert PIL to numpy (GFPGAN works with numpy arrays)
            img_np = np.array(image)
            
            # Restore faces
            _, _, output = restorer.enhance(img_np, has_aligned=False, only_center_face=False, pad=10, weight=0.7)
            
            # Convert back to PIL
            restored = Image.fromarray(output)
            
            print("[LumaForgePipeline] ✅ Face restoration completed with GFPGAN")
            return restored
        except Exception as e:
            print(f"[LumaForgePipeline Warning] Face restoration failed ({e}). Continuing without restoration.")
            return image

    def _upscale_image(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """
        Upscales image using Real-ESRGAN for maximum clarity and detail.
        Falls back to Lanczos if Real-ESRGAN unavailable.
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Initialize Real-ESRGAN
            upsampler = RealESRGANer(
                scale=scale,
                model_name='RealESRGAN_x2plus',
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "mps" else False
            )
            
            # Convert PIL to numpy
            img_np = np.array(image)
            
            # Upscale
            output, _ = upsampler.enhance(img_np, outscale=scale)
            
            # Convert back to PIL
            upscaled = Image.fromarray(output)
            
            print(f"[LumaForgePipeline] ✅ Image upscaled {scale}x with Real-ESRGAN")
            return upscaled
        except Exception as e:
            print(f"[LumaForgePipeline] Real-ESRGAN unavailable ({e}). Using Lanczos upscaling.")
            new_size = (image.width * scale, image.height * scale)
            return image.resize(new_size, Image.Resampling.LANCZOS)

    def _enhance_clarity(self, image: Image.Image) -> Image.Image:
        """
        Enhances image clarity through multiple post-processing techniques.
        """
        # 1. Unsharp mask for edge enhancement
        blurred = image.filter(ImageFilter.GaussianBlur(1.0))
        img_arr = np.array(image, dtype=float)
        blur_arr = np.array(blurred, dtype=float)
        unsharp_mask = img_arr - blur_arr
        
        enhanced_arr = img_arr + 0.5 * unsharp_mask
        enhanced_arr = np.clip(enhanced_arr, 0, 255).astype(np.uint8)
        enhanced = Image.fromarray(enhanced_arr)
        
        # 2. Contrast boost
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.1)
        
        # 3. Sharpness boost
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness_enhancer.enhance(1.2)
        
        print("[LumaForgePipeline] ✅ Clarity enhancement applied")
        return enhanced

    def _generate_mock_image(self, prompt: str, width: int, height: int, aspect_ratio: str, seed: int) -> Image:
        """
        Generates a beautiful, highly stylized mock image dynamically matching the prompt.
        Draws a detailed cyberpunk retro-wave sci-fi landscape (stars, glowing sun,
        perspective grid, silhouette mountains) instead of a plain gradient,
        providing a stunning visual output in mock mode.
        """
        import math
        random.seed(seed)
        
        # 1. Base gradient colors based on prompt content
        colors = self._determine_colors_from_prompt(prompt)
        c1, c2 = colors[0], colors[1]
        
        # Create base canvas
        base = Image.new("RGB", (width, height), c1)
        draw = ImageDraw.Draw(base)
        
        # Draw vertical background gradient
        for y in range(height):
            ratio = y / height
            r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
            g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
            b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Draw starfield in the sky (top half of image)
        num_stars = random.randint(30, 60)
        for _ in range(num_stars):
            sx = random.randint(0, width)
            sy = random.randint(0, int(height * 0.6))
            star_size = random.choice([1, 2, 3])
            star_alpha = random.randint(100, 255)
            if star_size > 1:
                draw.line([(sx - star_size, sy), (sx + star_size, sy)], fill=(255, 255, 255, star_alpha))
                draw.line([(sx, sy - star_size), (sx, sy + star_size)], fill=(255, 255, 255, star_alpha))
            else:
                draw.point((sx, sy), fill=(255, 255, 255, star_alpha))

        # 2. Draw a glowing retro sun/planet
        sun_r = int(min(width, height) * 0.22)
        sun_cx = random.randint(int(width * 0.3), int(width * 0.7))
        sun_cy = int(height * 0.45)
        
        sun_color = c2
        if len(colors) > 2:
            sun_color = colors[2]
        else:
            sun_color = (255, 110, 0) if "fire" in prompt.lower() else (255, 0, 127)
            
        for r_step in range(sun_r, 0, -2):
            glow_alpha = int(80 * (1 - r_step / sun_r))
            draw.ellipse(
                [sun_cx - r_step, sun_cy - r_step, sun_cx + r_step, sun_cy + r_step],
                outline=(sun_color[0], sun_color[1], sun_color[2], glow_alpha),
                width=2
            )
            
        slice_height = 6
        gap_height = 3
        for y_offset in range(-sun_r, sun_r):
            x_half = int(math.sqrt(max(0, sun_r**2 - y_offset**2)))
            current_y = sun_cy + y_offset
            
            # Retro scanline gaps at the bottom of the sun
            if y_offset > 0 and (y_offset // (slice_height + gap_height)) % 2 == 0:
                continue
                
            draw.line([(sun_cx - x_half, current_y), (sun_cx + x_half, current_y)], fill=sun_color)

        # 3. Draw Cyberpunk Perspective Grid (Ground)
        horizon_y = int(height * 0.55)
        grid_color = (0, 242, 254) if "neon" in prompt.lower() or "cyberpunk" in prompt.lower() else (255, 255, 255)
        
        # Horizontal lines getting closer as they approach the horizon
        num_grid_lines = 12
        for i in range(num_grid_lines):
            t = i / (num_grid_lines - 1)
            line_y = int(horizon_y + (height - horizon_y) * (t ** 2.2))
            alpha = int(40 + 180 * t)
            draw.line([(0, line_y), (width, line_y)], fill=(grid_color[0], grid_color[1], grid_color[2], alpha), width=1)
            
        # Vanishing lines
        num_vanishing_lines = 16
        vanishing_cx = width // 2
        for i in range(num_vanishing_lines):
            t = i / (num_vanishing_lines - 1)
            start_x = int(width * (t * 2 - 0.5))
            alpha = int(120)
            draw.line([(vanishing_cx, horizon_y), (start_x, height)], fill=(grid_color[0], grid_color[1], grid_color[2], alpha), width=1)

        # 4. Draw Silhouette mountains at the horizon
        num_mountains = 3
        for idx in range(num_mountains):
            m_points = []
            m_width = random.randint(int(width * 0.4), int(width * 0.8))
            m_height = random.randint(40, 90)
            m_cx = random.randint(0, width)
            
            m_points.append((m_cx - m_width // 2, horizon_y))
            m_points.append((m_cx - m_width // 4, horizon_y - m_height // 2))
            m_points.append((m_cx, horizon_y - m_height))
            m_points.append((m_cx + m_width // 4, horizon_y - m_height // 3))
            m_points.append((m_cx + m_width // 2, horizon_y))
            
            draw.polygon(m_points, fill=(10, 10, 25))

        # Composite base with overlays
        image = base.convert("RGB")
        
        # 5. Enhance detail and sharpness across the entire mock canvas (no edge blurs)
        image = image.filter(ImageFilter.SHARPEN)
        
        # 6. Add minimalist border frame
        draw_frame = ImageDraw.Draw(image, "RGBA")
        draw_frame.rectangle([15, 15, width - 15, height - 15], outline=(255, 255, 255, 60), width=2)
        
        # Metadata printouts
        settings_text = f"SEED: {seed} | RES: {width}x{height} | {aspect_ratio} | MOCK ENGINE"
        draw_frame.text((30, height - 48), settings_text, fill=(255, 255, 255, 140))
        
        title_limit = prompt[:38] + ("..." if len(prompt) > 38 else "")
        draw_frame.text((30, height - 68), f"PROMPT: {title_limit.upper()}", fill=(255, 255, 255, 220))
        
        return image

    def _determine_colors_from_prompt(self, prompt: str) -> list:
        """Determines color palette based on keywords in the prompt."""
        p = prompt.lower()
        
        palettes = {
            "cyberpunk": [(26, 8, 46), (255, 0, 127)],    # Dark purple -> hot pink
            "neon": [(10, 25, 47), (0, 242, 254)],       # Deep blue -> bright cyan
            "fire": [(40, 10, 5), (255, 110, 0)],         # Dark red -> intense orange
            "forest": [(10, 30, 20), (46, 204, 113)],     # Deep green -> emerald
            "cosmic": [(11, 11, 28), (142, 68, 173)],     # Starry indigo -> amethyst
            "sunset": [(230, 81, 0), (253, 216, 53)],     # Deep orange -> amber gold
            "character": [(20, 20, 20), (140, 140, 150)], # Studio grey -> soft silver
            "poster": [(10, 15, 25), (241, 196, 15)]      # Dark blue-gray -> poster gold
        }
        
        for key, palette in palettes.items():
            if key in p:
                return palette
                
        # Default harmonized cool blue gradient
        return [(15, 32, 67), (70, 130, 180)]

    def _overlay_poster_typography(self, image: Image, title: str) -> Image:
        """Overlays professional premium typography on the generated movie poster image."""
        try:
            from PIL import ImageDraw, ImageFont, ImageFilter, ImageOps
            import os
            import re
            
            # Copy base canvas
            img = image.copy()
            width, height = img.size
            
            # Clean title
            title_text = title.strip().upper()
            
            # Detect layout style from prompt/title text
            style_type = "cinematic"
            if any(w in title_text.lower() for w in ["cyber", "neon", "retro", "hack", "system", "matrix", "future", "laser", "star", "cosmic", "galaxy"]):
                style_type = "scifi"
            elif any(w in title_text.lower() for w in ["luxury", "gold", "royal", "silent", "whisper", "minimal", "white", "glass", "vogue", "velvet"]):
                style_type = "luxury"
                
            # Helper for character-spaced drawing
            def get_spaced_text_width(text, font, spacing=6):
                w = 0
                for char in text:
                    bbox = font.getbbox(char)
                    char_w = bbox[2] - bbox[0]
                    w += char_w + spacing
                return w - spacing if w > 0 else 0

            def draw_spaced_text(draw, position, text, font, fill, spacing=6, shadow_fill=None, shadow_offset=(1, 1)):
                x, y = position
                ox, oy = shadow_offset
                for char in text:
                    if shadow_fill:
                        draw.text((x + ox, y + oy), char, fill=shadow_fill, font=font)
                    draw.text((x, y), char, fill=fill, font=font)
                    bbox = font.getbbox(char)
                    char_w = bbox[2] - bbox[0]
                    x += char_w + spacing

            def draw_gradient_text(target_img, position, text, font, spacing, top_color, bottom_color, shadow_fill=None, shadow_offset=(2, 2)):
                """Draws text with a beautiful top-to-bottom vertical color gradient."""
                w = get_spaced_text_width(text, font, spacing)
                bbox = font.getbbox("A")
                h = bbox[3] - bbox[1] + 15
                
                # Create a mask for the text
                mask = Image.new("L", (w + 40, h + 20), 0)
                mask_draw = ImageDraw.Draw(mask)
                
                # Draw spaced text on mask
                x_m, y_m = 20, 10
                for char in text:
                    mask_draw.text((x_m, y_m), char, fill=255, font=font)
                    c_bbox = font.getbbox(char)
                    char_w = c_bbox[2] - c_bbox[0]
                    x_m += char_w + spacing
                    
                # Create gradient image of the same size
                gradient = Image.new("RGBA", (w + 40, h + 20))
                g_draw = ImageDraw.Draw(gradient)
                for y in range(h + 20):
                    ratio = y / (h + 20)
                    r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
                    g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
                    b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
                    g_draw.line([(0, y), (w + 40, y)], fill=(r, g, b, 255))
                    
                # Apply mask to gradient
                text_img = Image.new("RGBA", (w + 40, h + 20))
                text_img.paste(gradient, (0, 0), mask)
                
                # Draw shadow on the main image if requested
                if shadow_fill:
                    sx, sy = position[0] + shadow_offset[0], position[1] + shadow_offset[1]
                    shadow_img = Image.new("RGBA", (w + 40, h + 20), (shadow_fill[0], shadow_fill[1], shadow_fill[2], shadow_fill[3]))
                    target_img.paste(shadow_img, (sx - 20, sy - 10), mask)
                    
                # Paste onto main image
                target_img.paste(text_img, (position[0] - 20, position[1] - 10), mask)

            # Setup fonts based on theme
            font_paths = {
                "scifi": "/System/Library/Fonts/Supplemental/Futura.ttc",
                "luxury": "/System/Library/Fonts/Supplemental/Didot.ttc",
                "cinematic": "/System/Library/Fonts/Supplemental/Copperplate.ttc"
            }
            sub_font_paths = {
                "scifi": "/System/Library/Fonts/Supplemental/Futura.ttc",
                "luxury": "/System/Library/Fonts/Supplemental/Baskerville.ttc",
                "cinematic": "/System/Library/Fonts/Supplemental/Georgia.ttf"
            }
            
            # Select active fonts with Helvetica fallbacks
            font_path = font_paths.get(style_type, "/System/Library/Fonts/Helvetica.ttc")
            sub_font_path = sub_font_paths.get(style_type, "/System/Library/Fonts/Helvetica.ttc")
            
            if not os.path.exists(font_path):
                font_path = "/System/Library/Fonts/Helvetica.ttc"
            if not os.path.exists(sub_font_path):
                sub_font_path = "/System/Library/Fonts/Helvetica.ttc"
                
            # Font size heuristics
            title_font_size = max(26, int(height * 0.08))
            sub_font_size = max(10, int(height * 0.024))
            credits_font_size = max(8, int(height * 0.016))
            
            # Determine maximum allowable width
            max_w = int(width * 0.88)
            
            try:
                t_font = ImageFont.truetype(font_path, title_font_size)
                # Compute width with spacing (default spacing is 8 for title)
                t_spacing = 8 if style_type != "luxury" else 14
                t_w = get_spaced_text_width(title_text, t_font, spacing=t_spacing)
                
                # Shrink title if too wide
                while t_w > max_w and title_font_size > 16:
                    title_font_size -= 2
                    t_font = ImageFont.truetype(font_path, title_font_size)
                    t_w = get_spaced_text_width(title_text, t_font, spacing=t_spacing)
            except Exception:
                t_font = ImageFont.load_default()
                t_spacing = 4
                t_w = len(title_text) * (8 + t_spacing)
                
            # Create overlay canvas
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            
            if style_type == "scifi":
                # 1. Cyberpunk/Sci-Fi Theme
                # Bottom vignette (cyan/dark)
                for y in range(int(height * 0.6), height):
                    ratio = (y - int(height * 0.6)) / (height * 0.4)
                    alpha = int(210 * (ratio ** 1.5))
                    draw_line = ImageDraw.Draw(overlay)
                    draw_line.line([(0, y), (width, y)], fill=(5, 10, 20, alpha))
                
                # Draw Title at the bottom with gradient
                tx = (width - t_w) // 2
                ty = int(height * 0.82)
                
                draw_gradient_text(
                    overlay, (tx, ty), title_text, t_font, spacing=t_spacing,
                    top_color=(0, 255, 255), bottom_color=(0, 128, 255),
                    shadow_fill=(255, 0, 128, 200), shadow_offset=(-2, 2)
                )
                
                # Tagline / Subtitle
                draw_overlay = ImageDraw.Draw(overlay)
                sub_text = "A U R A _ G E N   //   N E T _ S Y S _ A C T I V E"
                try:
                    s_font = ImageFont.truetype(sub_font_path, sub_font_size)
                    s_w = get_spaced_text_width(sub_text, s_font, spacing=3)
                except Exception:
                    s_font = ImageFont.load_default()
                    s_w = len(sub_text) * 10
                sx = (width - s_w) // 2
                sy = int(height * 0.76)
                draw_spaced_text(draw_overlay, (sx, sy), sub_text, s_font, fill=(0, 240, 255, 220), spacing=3, shadow_fill=(0, 0, 0, 180))
                
                # Top coordinates HUD
                hud_text = "COORD: 35.6762° N, 139.6503° E | SYS: ONLINE"
                try:
                    h_font = ImageFont.truetype(sub_font_path, int(credits_font_size * 0.9))
                except Exception:
                    h_font = ImageFont.load_default()
                draw_overlay.text((30, 30), hud_text, fill=(0, 255, 255, 120), font=h_font)
                
            elif style_type == "luxury":
                # 2. Minimalist Luxury Theme
                # Top vignette (subtle dark vignette at top)
                for y in range(0, int(height * 0.35)):
                    ratio = 1.0 - (y / (height * 0.35))
                    alpha = int(140 * (ratio ** 1.8))
                    draw_line = ImageDraw.Draw(overlay)
                    draw_line.line([(0, y), (width, y)], fill=(8, 8, 12, alpha))
                    
                # Title at the top center with pearl gradient
                tx = (width - t_w) // 2
                ty = int(height * 0.15)
                
                draw_gradient_text(
                    overlay, (tx, ty), title_text, t_font, spacing=t_spacing,
                    top_color=(255, 255, 255), bottom_color=(235, 235, 240),
                    shadow_fill=(0, 0, 0, 180), shadow_offset=(2, 2)
                )
                
                # Gold separator line under title
                draw_overlay = ImageDraw.Draw(overlay)
                line_y = ty + int(height * 0.09)
                line_w = int(t_w * 0.6)
                lx1 = (width - line_w) // 2
                lx2 = lx1 + line_w
                draw_overlay.line([(lx1, line_y), (lx2, line_y)], fill=(212, 175, 55, 180), width=1) # gold line
                
                # Elegant tagline
                sub_text = "L U M A F O R G E   P R E S E N T S"
                try:
                    s_font = ImageFont.truetype(sub_font_path, int(sub_font_size * 0.95))
                    # Make it italic if Baskerville
                    if "Baskerville" in sub_font_path:
                        s_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Baskerville.ttc", int(sub_font_size * 0.95), index=1)
                    s_w = get_spaced_text_width(sub_text, s_font, spacing=4)
                except Exception:
                    s_font = ImageFont.load_default()
                    s_w = len(sub_text) * 10
                sx = (width - s_w) // 2
                sy = ty - int(height * 0.05)
                draw_spaced_text(draw_overlay, (sx, sy), sub_text, s_font, fill=(212, 175, 55, 220), spacing=4, shadow_fill=(0, 0, 0, 160), shadow_offset=(1, 1))
                
            else:
                # 3. Cinematic Action Theme (Default)
                # Bottom vignette (dark rich vignette)
                for y in range(int(height * 0.52), height):
                    ratio = (y - int(height * 0.52)) / (height * 0.48)
                    alpha = int(230 * (ratio ** 2.0))
                    draw_line = ImageDraw.Draw(overlay)
                    draw_line.line([(0, y), (width, y)], fill=(4, 4, 6, alpha))
                    
                # Title at bottom with warm silver/gold metallic gradient
                tx = (width - t_w) // 2
                ty = int(height * 0.80)
                
                draw_gradient_text(
                    overlay, (tx, ty), title_text, t_font, spacing=t_spacing,
                    top_color=(255, 255, 255), bottom_color=(220, 215, 200),
                    shadow_fill=(0, 0, 0, 245), shadow_offset=(3, 3)
                )
                
                # Dynamic billing block text (credits line)
                draw_overlay = ImageDraw.Draw(overlay)
                credits_line = "STARRING GENERATIVE IMAGINATION  •  EXECUTIVE PRODUCERS LUMAFORGE LABS  •  MUSIC BY NEURAL SYNTH"
                try:
                    c_font = ImageFont.truetype(font_path, credits_font_size)
                    c_w = get_spaced_text_width(credits_line, c_font, spacing=2)
                    # Shrink if too wide
                    while c_w > max_w and credits_font_size > 6:
                        credits_font_size -= 1
                        c_font = ImageFont.truetype(font_path, credits_font_size)
                        c_w = get_spaced_text_width(credits_line, c_font, spacing=2)
                except Exception:
                    c_font = ImageFont.load_default()
                    c_w = len(credits_line) * 8
                cx_pos = (width - c_w) // 2
                cy_pos = int(height * 0.90)
                draw_spaced_text(draw_overlay, (cx_pos, cy_pos), credits_line, c_font, fill=(160, 160, 160, 200), spacing=2)
                
                # Tagline above title
                tagline = "THE FUTURE OF CREATIVE ARTISTRY"
                try:
                    s_font = ImageFont.truetype(sub_font_path, sub_font_size)
                    # Make it italic if Georgia
                    if "Georgia" in sub_font_path:
                        s_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Italic.ttf", sub_font_size)
                    s_w = get_spaced_text_width(tagline, s_font, spacing=3)
                except Exception:
                    s_font = ImageFont.load_default()
                    s_w = len(tagline) * 10
                sx = (width - s_w) // 2
                sy = ty - int(height * 0.06)
                draw_spaced_text(draw_overlay, (sx, sy), tagline, s_font, fill=(225, 225, 225, 255), spacing=3, shadow_fill=(0, 0, 0, 200))
                
                # Small minimalist line
                line_y = (ty + sy + int(height * 0.02)) // 2
                line_w = int(width * 0.35)
                lx1 = (width - line_w) // 2
                lx2 = lx1 + line_w
                draw_overlay.line([(lx1, line_y), (lx2, line_y)], fill=(255, 255, 255, 70), width=1)

            # Convert base image to RGBA, composite overlay, convert back to RGB
            img_rgba = img.convert("RGBA")
            composited = Image.alpha_composite(img_rgba, overlay)
            
            return composited.convert("RGB")
        except Exception as e:
            print(f"[LumaForgePipeline Warning] Failed to overlay premium typography: {e}")
            return image

    def _overlay_lumaforge_logo(self, image: Image) -> Image:
        """
        Overlays a beautiful, premium, glassmorphic LumaForge brand logo badge
        in the bottom-right corner of the image to make it stand out extremely clearly.
        """
        try:
            from PIL import ImageDraw, ImageFont
            img = image.copy()
            
            # Create a separate RGBA draw layer for glassmorphism transparency blending
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            width, height = img.size
            
            # Badge dimensions & positioning (Bottom-Right corner)
            badge_w = 165
            badge_h = 32
            padding = 20
            
            x1 = width - padding - badge_w
            y1 = height - padding - badge_h
            x2 = width - padding
            y2 = height - padding
            
            # 1. Draw Glassmorphic Badge Plate (with fallback if rounded_rectangle fails)
            badge_fill = (10, 10, 16, 190)    # Sleek dark translucent plate
            badge_border = (255, 255, 255, 45) # Subtle white glass border
            
            try:
                draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=badge_fill, outline=badge_border, width=1)
            except AttributeError:
                # Fallback to standard rectangle if Pillow version is old
                draw.rectangle([x1, y1, x2, y2], fill=badge_fill, outline=badge_border, width=1)
            
            # Load system font for crisp, premium text rendering
            font_path = "/System/Library/Fonts/Helvetica.ttc"
            if not os.path.exists(font_path):
                font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
                
            try:
                font = ImageFont.truetype(font_path, 10)
            except Exception:
                font = ImageFont.load_default()
                
            # Minimalist spaced logo text
            text = "L U M A F O R G E"
            
            # 2. Draw brand text inside the badge
            text_x = x1 + 14
            # Center the text vertically
            text_y = y1 + (badge_h // 2) - 6
            
            # Draw subtle text drop shadow
            draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0, 180), font=font)
            draw.text((text_x, text_y), text, fill=(255, 255, 255, 240), font=font)
            
            # 3. Draw geometric diamond forge logo next to it
            diamond_w = 12
            diamond_h = 12
            center_x = x2 - 20
            center_y = y1 + (badge_h // 2)
            
            diamond_points = [
                (center_x, center_y - diamond_h//2),
                (center_x + diamond_w//2, center_y),
                (center_x, center_y + diamond_h//2),
                (center_x - diamond_w//2, center_y)
            ]
            
            # Diamond shadow
            shadow_points = [(x + 1, y + 1) for x, y in diamond_points]
            draw.polygon(shadow_points, fill=(0, 0, 0, 180))
            draw.polygon(diamond_points, fill=(255, 255, 255, 240))
            
            # Inner dark core accent cut
            inner_w = 4
            inner_h = 4
            inner_points = [
                (center_x, center_y - inner_h//2),
                (center_x + inner_w//2, center_y),
                (center_x, center_y + inner_h//2),
                (center_x - inner_w//2, center_y)
            ]
            draw.polygon(inner_points, fill=(10, 10, 16, 255))
            
            # Alpha composite the overlay layer onto the original image
            final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
            return final_img.convert("RGB")
            
        except Exception as e:
            print(f"[LumaForgePipeline Warning] Failed to overlay watermark logo badge: {e}")
            return image

    def colorize(self, image: Image.Image, style: str = "vibrant", mock: bool = False) -> dict:
        """
        Colorizes an image with different color grading styles.
        Styles: vibrant, warm, cool, vintage, sepia
        """
        start_time = time.time()
        start_mem_bytes = self._get_mps_memory()
        
        result = image.convert("RGB").copy()
        
        if style == "vibrant":
            # Boost saturation and contrast
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.6)
            contrast = ImageEnhance.Contrast(result)
            result = contrast.enhance(1.2)
        elif style == "warm":
            # Warm color temperature shift
            r, g, b = result.split()
            r = r.point(lambda x: min(255, int(x * 1.15)))
            g = g.point(lambda x: int(x * 0.95))
            result = Image.merge("RGB", (r, g, b))
        elif style == "cool":
            # Cool color temperature shift
            r, g, b = result.split()
            r = r.point(lambda x: int(x * 0.85))
            b = b.point(lambda x: min(255, int(x * 1.15)))
            result = Image.merge("RGB", (r, g, b))
        elif style == "vintage":
            # Vintage film look
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(0.8)
            result = result.convert("RGBA")
            overlay = Image.new("RGBA", result.size, (255, 200, 100, 30))
            result = Image.alpha_composite(result, overlay).convert("RGB")
        elif style == "sepia":
            # Classic sepia tone
            result = result.convert("LA").convert("RGB")
            r, g, b = result.split()
            r = r.point(lambda x: min(255, int(x * 1.2)))
            g = g.point(lambda x: int(x * 0.95))
            b = b.point(lambda x: int(x * 0.7))
            result = Image.merge("RGB", (r, g, b))
        
        latency_sec = time.time() - start_time
        end_mem_bytes = self._get_mps_memory()
        memory_used_mb = max(0.0, (end_mem_bytes - start_mem_bytes) / (1024 * 1024))
        
        return {
            "image": result,
            "latency_sec": latency_sec,
            "memory_used_mb": memory_used_mb,
            "style": style
        }

    def restore_face(self, image: Image.Image, intensity: str = "medium", mock: bool = False) -> dict:
        """
        Restores and enhances facial features.
        Intensity levels: low, medium, high, ultra
        """
        start_time = time.time()
        start_mem_bytes = self._get_mps_memory()
        
        result = image.convert("RGB").copy()
        
        # Intensity mapping for enhancement factors
        intensity_map = {
            "low": 1.1,
            "medium": 1.3,
            "high": 1.5,
            "ultra": 1.8
        }
        factor = intensity_map.get(intensity, 1.3)
        
        # Step 1: Denoise
        result = result.filter(ImageFilter.MedianFilter(size=3))
        
        # Step 2: Sharpen
        sharpness = ImageEnhance.Sharpness(result)
        result = sharpness.enhance(factor)
        
        # Step 3: Enhance contrast
        contrast = ImageEnhance.Contrast(result)
        result = contrast.enhance(factor * 0.7)
        
        # Step 4: Boost color vibrancy for skin tones
        color = ImageEnhance.Color(result)
        result = color.enhance(1.15)
        
        latency_sec = time.time() - start_time
        end_mem_bytes = self._get_mps_memory()
        memory_used_mb = max(0.0, (end_mem_bytes - start_mem_bytes) / (1024 * 1024))
        
        return {
            "image": result,
            "latency_sec": latency_sec,
            "memory_used_mb": memory_used_mb,
            "intensity": intensity
        }

