import json
import urllib.request
import urllib.error

class OllamaClient:
    def __init__(self, host="http://localhost:11434", model="llama3.2:1b"):
        self.host = host.rstrip('/')
        self.model = model
        print(f"[OllamaClient] Initialized with model: {self.model}")

    def _call_api(self, endpoint, data):
        url = f"{self.host}{endpoint}"
        req_data = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            # If Ollama is offline or times out, return None
            print(f"[OllamaClient Warning] Failed to connect to Ollama (using fallback): {type(e).__name__}")
            return None

    def check_connection(self):
        """Check if Ollama is running and responsive."""
        data = {"model": self.model, "prompt": "test", "stream": False}
        res = self._call_api("/api/generate", data)
        is_connected = res is not None
        if is_connected:
            print("[OllamaClient] ✅ Connected to Ollama server")
        else:
            print("[OllamaClient] ⚠️ Ollama not available - using heuristic fallbacks")
        return is_connected

    def classify_safety(self, prompt: str) -> dict:
        """
        Classifies a user prompt into SAFE, BORDERLINE, or UNSAFE.
        """
        system_instruction = (
            "You are a strict safety classifier for a text-to-image generation system. "
            "Your task is to classify the user's prompt into one of three categories:\n"
            "- SAFE: The prompt is completely safe and appropriate.\n"
            "- BORDERLINE: The prompt contains slightly sensitive concepts (like horror, fantasy weapons, mild combat, or suggestive words in a non-explicit context) but is acceptable if rewritten slightly or handled with care.\n"
            "- UNSAFE: The prompt contains explicit violence, illegal activities, hate speech, severe gore, or explicit NSFW content.\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            '{"classification": "SAFE" | "BORDERLINE" | "UNSAFE", "reason": "A brief 1-sentence reason"}'
        )

        data = {
            "model": self.model,
            "prompt": f"{system_instruction}\n\nPrompt to classify: \"{prompt}\"\n\nJSON output:",
            "stream": False,
            "format": "json"
        }

        res = self._call_api("/api/generate", data)
        if not res:
            # Fallback heuristic if Ollama is offline
            return self._heuristic_classify_safety(prompt)

        try:
            content = res.get("response", "").strip()
            return json.loads(content)
        except Exception:
            return {"classification": "SAFE", "reason": "Failed to parse Ollama response, defaulting to SAFE."}

    def _heuristic_classify_safety(self, prompt: str) -> dict:
        """Simple keyword fallback classifier when Ollama is offline."""
        unsafe_keywords = ["nsfw", "naked", "porn", "gore", "kill", "murder", "bomb", "suicide", "terrorist", "torture"]
        borderline_keywords = ["fight", "blood", "sword", "weapon", "monster", "vampire", "scary", "ghost", "darkness"]
        
        prompt_lower = prompt.lower()
        for kw in unsafe_keywords:
            if kw in prompt_lower:
                return {"classification": "UNSAFE", "reason": f"Prompt flagged by keyword check: '{kw}'."}
        
        for kw in borderline_keywords:
            if kw in prompt_lower:
                return {"classification": "BORDERLINE", "reason": f"Prompt marked as borderline due to keyword: '{kw}'."}
                
        return {"classification": "SAFE", "reason": "Local keyword checks passed."}

    def rewrite_prompt(self, prompt: str) -> str:
        """
        Rewrites a borderline prompt to remove sensitive elements while retaining the core creative vision.
        """
        system_instruction = (
            "You are a helpful prompt refiner. Your task is to rewrite a sensitive or borderline text-to-image prompt "
            "to make it safe, constructive, and appropriate while preserving the core creative idea. "
            "Remove any blood, gore, excessive horror, or suggestive elements, and replace them with dramatic style, "
            "heroic aesthetics, or stylized fantasy concepts. Keep your response extremely brief, returning ONLY the rewritten prompt."
        )

        data = {
            "model": self.model,
            "prompt": f"{system_instruction}\n\nOriginal prompt: \"{prompt}\"\n\nRewritten prompt:",
            "stream": False
        }

        res = self._call_api("/api/generate", data)
        if not res:
            # Basic offline rewrite logic
            return prompt.replace("blood", "red paint").replace("gore", "intensity").replace("kill", "defeat")

        rewritten = res.get("response", "").strip().strip('"').strip("'")
        
        # Check if the rewritten response is an LLM refusal (false positive safety trigger)
        low_rewritten = rewritten.lower()
        refusal_markers = [
            "sorry", "fulfill", "request", "cannot", "can't", "guidelines",
            "policy", "inappropriate", "unable to", "restrict", "violation"
        ]
        
        if not rewritten or any(marker in low_rewritten for marker in refusal_markers):
            print(f"[OllamaClient Warning] Rewrite failed/refused (returned: '{rewritten}'). Using heuristic fallback.")
            clean_prompt = prompt
            replacements = {
                "blood": "red paint",
                "gore": "intensity",
                "kill": "defeat",
                "dead": "fallen",
                "murder": "defeat",
                "suicide": "sacrifice",
                "naked": "dressed",
                "nude": "dressed",
                "porn": "fine art",
                "terrorist": "warrior",
                "bomb": "crystal energy"
            }
            for word, rep in replacements.items():
                import re
                clean_prompt = re.sub(re.escape(word), rep, clean_prompt, flags=re.IGNORECASE)
            return clean_prompt

        return rewritten

    def expand_prompt(self, prompt: str, mode: str = "general", category: str = None, subcategory: str = None) -> dict:
        """
        Expands the user prompt using predefined style presets and category descriptors.
        """
        import re
        
        scene_desc = prompt.strip()
        
        mode_prompts = {
            "art": "digital concept art, highly detailed, fantasy sci-fi surreal elements, matte painting style, vivid colors, masterfully rendered",
            "character": "detailed character design, face close-up, full body view, character portrait, high resolution features, realistic proportions",
            "landscape": "scenic landscape, natural scenery, epic vistas, 8k resolution, volumetric atmosphere, detailed clouds, beautiful natural lighting",
            "architecture": "architectural photography, modern building exterior, luxury high-end interior, raytraced reflection, sharp lines, cinematic design",
            "vehicle": "sleek sports car automotive photography, dynamic reflections, glossy metallic paint, dramatic lighting, sharp focus on chassis",
            "product": "studio product mockup design, professional commercial advertising, clean product lighting, soft white backdrop, elegant minimalist packaging",
            "marketing": "marketing poster design, commercial branding graphics, bold colors, professional graphic design layout, vector advertising poster",
            "food": "appetizing gourmet food plating photography, close-up delicious shot, professional food styling, organic fresh ingredients, warm lighting, blurred background",
            "fashion": "high fashion lookbook editorial photography, designer clothing, haute couture runway style, model posing, dramatic studio lighting",
            "game": "fantasy game asset, detailed icon, weapon sprite, interface vector, dark clean background, isolated graphic, item artifact",
            "animal": "national geographic wildlife photography, sharp animal portrait, detailed fur textures, macro focus on eyes, natural habitat background",
            "event": "elegant festival poster design, celebration event invitation artwork, bright colors, greeting card design",
            "business": "flat vector illustration, corporate infographic chart style, clean business graphics, presentation design elements, modern company colors",
            "education": "clean scientific textbook illustration, medical biology schema diagram, detailed educational graphics, clear pointers and arrows",
            "style_anime": "vibrant anime key visual style, highly detailed digital illustration, cel shaded, anime sketch, masterfully drawn",
            "style_sketch": "hand-drawn pencil sketch, fine graphite line shading, cross-hatching detail, white textured paper background",
            "style_oil": "oil on canvas art masterpiece, thick textured impasto brushstrokes, realistic paint texture, museum lighting",
            "style_pixel": "retro pixel art, 8-bit game console graphics, 16-bit arcade sprite aesthetic, pixelated texture, vintage gaming",
            "style_watercolor": "watercolor wash painting, delicate soft splatters, bleeding pastel pigment textures, hand-painted textured paper artwork"
        }
        
        if mode == "poster":
            quoted_titles = re.findall(r'["\']([^"\']+)["\']', prompt)
            if quoted_titles:
                title = quoted_titles[0]
                scene_desc = f'{prompt.strip()}, movie poster "{title}" with bold typography'
            else:
                scene_desc = f"{prompt.strip()}, cinematic movie poster layout"
        elif mode in mode_prompts:
            scene_desc = f"{prompt.strip()}, {mode_prompts[mode]}"
            
        # Prevent fusion artifacts by detailing vague 'holding' actions
        holding_pattern = re.compile(r'\b(holding|carrying|wielding|holding up|armed with)\b\s+(a|an|the)?\s*', re.IGNORECASE)
        holding_match = holding_pattern.search(scene_desc)
        if holding_match:
            if not any(kw in scene_desc.lower() for kw in ["hand", "grip", "hilt", "stance", "pose", "clutching", "brandishing", "raised", "wielding with"]):
                # Extract the noun phrase up to the next comma or end of string
                start_idx = holding_match.end()
                rest = scene_desc[start_idx:]
                comma_idx = rest.find(',')
                if comma_idx != -1:
                    noun_phrase = rest[:comma_idx].strip()
                    after_noun = rest[comma_idx:]
                else:
                    noun_phrase = rest.strip()
                    after_noun = ""
                
                # Build a detailed holding phrase
                # Determine appropriate grip description based on standard nouns
                if any(w in noun_phrase.lower() for w in ["sword", "weapon", "blade", "dagger", "saber", "axe", "staff", "shield", "spear", "lance", "gun", "pistol", "rifle"]):
                    detailed_hold = f"gripping the hilt and handle of the {noun_phrase} firmly in one hand, posing in a natural heroic stance"
                else:
                    detailed_hold = f"holding the {noun_phrase} firmly in their hand, posing naturally"
                
                scene_desc = scene_desc[:holding_match.start()] + detailed_hold + after_noun
            
        # Build response dict
        expanded = {
            "subject": scene_desc,
            "action": "",
            "environment": "",
            "style": mode_prompts.get(mode, ""),
            "lighting": "",
            "camera": "",
            "mood": "",
            "quality_emphasis": "8k resolution, masterfully rendered",
            "safety_constraints": "safe for work",
            "full_prompt": scene_desc
        }
        
        return expanded

    def optimize_prompt_for_sd35(self, prompt: str, max_tokens: int = 256) -> dict:
        """
        Uses Ollama iteratively to compress a prompt to fit SD 3.5 Medium's T5 token limit (256 tokens).
        Keeps trying with stricter instructions until successful.
        """
        # Estimate current tokens (rough: 1 token ≈ 1.3 chars)
        estimated_tokens = len(prompt) / 1.3
        
        if estimated_tokens <= max_tokens:
            # Already under limit, return as-is
            return {
                "optimized_prompt": prompt,
                "original_tokens": int(estimated_tokens),
                "final_tokens": int(estimated_tokens),
                "was_compressed": False
            }
        
        max_chars = int(max_tokens * 1.3)  # 256 tokens ≈ 332 chars
        optimized = prompt
        attempt = 0
        max_attempts = 3
        
        # Try iteratively with increasingly strict instructions
        while attempt < max_attempts:
            attempt += 1
            
            if attempt == 1:
                # First attempt: Gentle compression
                instruction = (
                    f"Compress this image prompt to MAXIMUM {max_chars} characters.\n"
                    f"Keep main subject, key details, lighting, style. Remove filler words.\n"
                    f"Use commas between concepts. Output ONLY the compressed prompt."
                )
            elif attempt == 2:
                # Second attempt: More aggressive
                instruction = (
                    f"URGENT: Compress to EXACTLY {max_chars} characters or LESS.\n"
                    f"Remove ALL: 'a', 'an', 'the', 'with', 'on', 'at', 'in', 'of'.\n"
                    f"Keep: subject, visuals, style. Use commas. NO extra words."
                )
            else:
                # Final attempt: Maximum compression
                instruction = (
                    f"CRITICAL: Must be {max_chars} chars MAX. Current too long.\n"
                    f"Only keep: main subject, 2-3 key adjectives, style, lighting.\n"
                    f"Format: 'subject, detail, detail, style, lighting' - nothing more."
                )
            
            data = {
                "model": self.model,
                "prompt": f"{instruction}\n\nInput ({len(optimized)} chars): \"{optimized}\"\n\nOutput:",
                "stream": False
            }
            
            res = self._call_api("/api/generate", data)
            if not res:
                print(f"[OllamaClient] Ollama unavailable, using heuristic fallback")
                return self._heuristic_compress_prompt(prompt, max_tokens)
            
            new_optimized = res.get("response", "").strip().strip('"').strip("'")
            
            # Validate compression
            if not new_optimized or len(new_optimized) >= len(optimized):
                print(f"[OllamaClient] Attempt {attempt}: Ollama didn't compress, retrying...")
                continue
            
            optimized = new_optimized
            final_tokens = len(optimized) / 1.3
            
            # Success! Check if under limit
            if final_tokens <= max_tokens and len(optimized) <= max_chars:
                print(f"[OllamaClient] ✅ Compressed successfully in {attempt} attempt(s): {int(estimated_tokens)} → {int(final_tokens)} tokens")
                return {
                    "optimized_prompt": optimized,
                    "original_tokens": int(estimated_tokens),
                    "final_tokens": int(final_tokens),
                    "was_compressed": True
                }
            else:
                print(f"[OllamaClient] Attempt {attempt}: {int(final_tokens)} tokens, still too long, retrying...")
        
        # After max attempts, use heuristic as last resort
        print(f"[OllamaClient] ⚠️  Failed after {max_attempts} attempts, using heuristic fallback")
        return self._heuristic_compress_prompt(prompt, max_tokens)
    
    def _heuristic_compress_prompt(self, prompt: str, max_tokens: int = 256) -> dict:
        """Aggressive fallback compression when Ollama is offline or doesn't compress enough."""
        import re
        
        estimated_original = len(prompt) / 1.3
        max_chars = int(max_tokens * 1.3)  # 256 tokens ≈ 332 chars
        
        # Step 1: Split into words and remove filler words aggressively
        fillers = {'a', 'an', 'the', 'with', 'in', 'at', 'on', 'of', 'and', 'or', 'but',
                   'very', 'extremely', 'really', 'quite', 'some', 'this', 'that', 
                   'is', 'are', 'was', 'were', 'being', 'been', 'be', 'has', 'have'}
        
        words = prompt.replace(',', ' ').split()
        essential_words = [w.strip('.,;:!?') for w in words if w.lower() not in fillers]
        
        # Step 2: Join with commas (more token-efficient than spaces for SD)
        compressed = ', '.join(essential_words)
        
        # Step 3: If still too long, truncate intelligently at word boundaries
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars]
            # Cut at last comma for clean break
            if ',' in compressed:
                compressed = compressed.rsplit(',', 1)[0].strip()
            else:
                compressed = compressed.rsplit(' ', 1)[0].strip()
        
        # Step 4: Final safety check - if STILL too long, hard truncate
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars-3].strip() + '...'
        
        estimated_final = len(compressed) / 1.3
        
        print(f"[OllamaClient] Heuristic compression: {len(prompt)} → {len(compressed)} chars ({int(estimated_original)} → {int(estimated_final)} tokens)")
        
        return {
            "optimized_prompt": compressed,
            "original_tokens": int(estimated_original),
            "final_tokens": int(estimated_final),
            "was_compressed": True
        }

    def check_prompt_coherence(self, prompt: str) -> dict:
        """
        Analyzes a prompt to ensure it obeys logical, physical, and scientific consistency.
        Returns a dictionary with coherence_score, level, violations, and recommendation.
        """
        system_instruction = (
            "You are a physics, logic, and spatial consistency checker for AI image generation prompts.\n"
            "Identify clear physical contradictions, scientific impossibilities, logic errors, or vague spatial/anatomical interactions (e.g. underwater fire, sunset at midnight, or 'holding/carrying' an object without describing the pose/grip/hands, which leads to body-object fusion glitches in diffusion models).\n"
            "If the prompt describes a physically possible scene with clear spatial and anatomy relationships, it is completely coherent (score 1.0, no violations).\n"
            "If the prompt has vague object interactions (e.g., 'holding a sword'), flag it as a violation/hazard and provide a recommendation to specify how they are holding/gripping it.\n"
            "Format your output ONLY as a JSON object with this exact structure:\n"
            "{\n"
            '  "coherence_score": 1.0 (if coherent) or 0.0 to 0.7 (if violations/hazards found),\n'
            '  "coherence_level": "high" (if score >= 0.8) or "medium" or "low",\n'
            '  "violations": ["list of issues/hazards found, or empty array if none"],\n'
            '  "recommendation": "rewritten prompt that enforces proper physics, structural logic, and specific posing, or empty string if already coherent and detailed",\n'
            '  "enhancement_needed": true | false\n'
            "}"
        )

        data = {
            "model": self.model,
            "prompt": f"{system_instruction}\n\nPrompt to evaluate: \"{prompt}\"\n\nJSON output:",
            "stream": False,
            "format": "json"
        }

        res = self._call_api("/api/generate", data)
        if not res:
            # Fallback heuristic if Ollama is offline
            return self._heuristic_check_coherence(prompt)

        try:
            content = res.get("response", "").strip()
            result = json.loads(content)
            # Ensure all required keys exist
            if "coherence_score" not in result:
                result["coherence_score"] = 0.85
            if "coherence_level" not in result:
                result["coherence_level"] = "high" if result["coherence_score"] > 0.8 else "medium"
            if "violations" not in result:
                result["violations"] = []
            if "recommendation" not in result:
                result["recommendation"] = ""
            if "enhancement_needed" not in result:
                result["enhancement_needed"] = len(result["violations"]) > 0
            return result
        except Exception:
            return self._heuristic_check_coherence(prompt)

    def _heuristic_check_coherence(self, prompt: str) -> dict:
        """Heuristic check when Ollama is offline."""
        violations = []
        p_lower = prompt.lower()
        
        # Check for lighting contradiction
        if "sunset" in p_lower and "noon" in p_lower:
            violations.append("Contradictory time of day: contains both 'sunset' and 'noon'.")
        if "neon light" in p_lower and "dark cave" in p_lower and not ("glowing" in p_lower or "illuminating" in p_lower):
            violations.append("Ambient lighting conflict: neon light in a dark cave needs explicit light emission description.")
            
        # Check for anatomy / physics contradiction
        if "floating" in p_lower and not any(kw in p_lower for kw in ["space", "zero gravity", "fantasy", "magic", "levitating", "flying"]):
            violations.append("Gravity violation: objects are 'floating' without space/fantasy context.")
        if "symmetrical asymmetry" in p_lower:
            violations.append("Semantic logic contradiction: 'symmetrical asymmetry'.")
            
        # Check for vague object interaction/holding which causes fusion artifacts
        import re
        holding_pattern = re.compile(r'\b(holding|carrying|wielding|holding up|armed with)\b\s+(a|an|the)?\s*', re.IGNORECASE)
        holding_match = holding_pattern.search(p_lower)
        if holding_match:
            if not any(kw in p_lower for kw in ["hand", "grip", "hilt", "stance", "pose", "clutching", "brandishing", "raised", "wielding with"]):
                # Extract noun phrase
                start_idx = holding_match.end()
                rest = p_lower[start_idx:]
                comma_idx = rest.find(',')
                if comma_idx != -1:
                    noun_phrase = rest[:comma_idx].strip()
                else:
                    noun_phrase = rest.strip()
                violations.append(
                    f"Vague interaction: '{holding_match.group(1)} {noun_phrase}' without specifying hand placement, grip, or pose. "
                    f"This frequently causes the image model to fuse the object into the character's body."
                )
            
        score = 1.0 - (len(violations) * 0.25)
        score = max(0.2, min(1.0, score))
        
        level = "high"
        if score < 0.6:
            level = "low"
        elif score < 0.85:
            level = "medium"
            
        recommendation = prompt
        if violations:
            # Basic recommendation fixing floating gravity
            if "floating" in p_lower and not any(kw in p_lower for kw in ["space", "zero-g", "magic"]):
                recommendation = f"{prompt}, realistically grounded in environment, subject to gravity"
            
            # Recommendation fixing vague holding
            holding_match_rec = holding_pattern.search(recommendation)
            if holding_match_rec and not any(kw in recommendation.lower() for kw in ["hand", "grip", "hilt", "stance", "pose"]):
                start_idx = holding_match_rec.end()
                rest = recommendation[start_idx:]
                comma_idx = rest.find(',')
                if comma_idx != -1:
                    noun_phrase = rest[:comma_idx].strip()
                    after_noun = rest[comma_idx:]
                else:
                    noun_phrase = rest.strip()
                    after_noun = ""
                
                # Determine appropriate grip description based on standard nouns
                if any(w in noun_phrase.lower() for w in ["sword", "weapon", "blade", "dagger", "saber", "axe", "staff", "shield", "spear", "lance", "gun", "pistol", "rifle"]):
                    detailed_hold = f"gripping the hilt and handle of the {noun_phrase} firmly in one hand, posing in a natural heroic stance"
                else:
                    detailed_hold = f"holding the {noun_phrase} firmly in their hand, posing naturally"
                
                recommendation = recommendation[:holding_match_rec.start()] + detailed_hold + after_noun
        
        return {
            "coherence_score": score,
            "coherence_level": level,
            "violations": violations,
            "recommendation": recommendation if violations else "",
            "enhancement_needed": len(violations) > 0
        }
