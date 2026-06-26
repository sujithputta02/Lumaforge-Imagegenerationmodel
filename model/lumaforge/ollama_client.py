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

        return res.get("response", "").strip().strip('"')

    def expand_prompt(self, prompt: str, mode: str = "general", category: str = None, subcategory: str = None) -> dict:
        """
        Expands a simple user prompt into a structured set of fields and a consolidated full prompt.
        Optionally integrates category-specific enhancements.
        """
        prompt_template = (
            "You are a prompt engineering assistant for the 'LumaForge' text-to-image model. "
            "Expand the user prompt into a detailed, structured prompt suited for high-quality image generation. "
            "Analyze the core request and structure it into these specific fields:\n"
            "- subject: The main character, object, or focus of the image.\n"
            "- action: What the subject is doing or their pose.\n"
            "- environment: The background setting, atmosphere, and surroundings.\n"
            "- style: The visual art style (e.g., cinematic, vector, 3D render, cyberpunk, fantasy illustration).\n"
            "- lighting: The light sources, direction, and intensity (e.g., dramatic backlighting, soft volumetric glow, neon contrast).\n"
            "- camera: The angle, lens, and focus depth (e.g., wide-angle cinematic shot, centered hero composition).\n"
            "- mood: The emotional tone of the scene (e.g., mysterious, heroic, ominous).\n"
            "- quality_emphasis: Terms to boost fidelity (e.g., highly detailed, polished finish).\n"
            "- safety_constraints: Guidelines to keep output appropriate.\n\n"
            f"Apply optimization rules for target mode: {mode.upper()}.\n"
            "If mode is POSTER: you MUST include: 'title-safe negative space at top and bottom, minimalist clean background, layout optimized for movie poster typography composition'.\n"
            "If mode is CHARACTER: emphasize detailed facial features, character sheets, action poses, and clean backgrounds.\n\n"
            "CRITICAL: Keep all field values extremely short and direct (1-3 words or brief phrases). "
            "Do NOT output nested dictionaries, lists, or key labels (like 'name:', 'keywords:') inside the JSON values. "
            "If the user prompt specifies any colors (e.g., 'red', 'blue', 'green', 'white'), you MUST explicitly preserve and reinforce those color descriptions in the 'subject' and 'style' fields.\n"
            "If the user prompt contains a movie title or text in quotes (e.g., 'Echoes of Mars'), you MUST preserve it exactly in quotes (e.g., \"Echoes of Mars\") in the 'subject' or 'style' field, and add typographic layout instructions like 'bold typography title text' to emphasize it.\n"
            "The entire combined prompt must be very concise (under 50 words total) to prevent token truncation by the image generator.\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "subject": "...",\n'
            '  "action": "...",\n'
            '  "environment": "...",\n'
            '  "style": "...",\n'
            '  "lighting": "...",\n'
            '  "camera": "...",\n'
            '  "mood": "...",\n'
            '  "quality_emphasis": "...",\n'
            '  "safety_constraints": "..."\n'
            "}"
        )

        data = {
            "model": self.model,
            "prompt": f"{prompt_template}\n\nUser prompt: \"{prompt}\"\n\nJSON output:",
            "stream": False,
            "format": "json"
        }

        res = self._call_api("/api/generate", data)
        
        fallback_fields = {
            "subject": prompt,
            "action": "standing",
            "environment": "simple background",
            "style": "cinematic movie poster" if mode == "poster" else "digital art character portrait",
            "lighting": "dramatic cinematic lighting",
            "camera": "centered hero shot",
            "mood": "heroic",
            "quality_emphasis": "high detail, polished finish",
            "safety_constraints": "artistic representation"
        }

        if not res:
            expanded = fallback_fields
        else:
            try:
                expanded = json.loads(res.get("response", "").strip())
            except Exception:
                expanded = fallback_fields

        # Fill in any missing keys
        for key, val in fallback_fields.items():
            if key not in expanded or not expanded[key]:
                expanded[key] = val

        # Sanitize and clean up the values
        import re
        def clean_val(val):
            if isinstance(val, dict):
                items = []
                for k, v in val.items():
                    if v:
                        items.append(clean_val(v))
                val = ", ".join(items)
            elif isinstance(val, list):
                val = ", ".join([clean_val(x) for x in val])
            
            val = str(val).strip()
            
            # Remove brackets, quotes, and structural prefixes (like "name: ", "description: ")
            val = re.sub(r'\b(name|description|type|keywords|style|lighting|camera|mood|subject|action|environment|quality_emphasis|safety_constraints)\s*:\s*', '', val, flags=re.IGNORECASE)
            val = val.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            val = re.sub(r'\s+', ' ', val)
            val = re.sub(r',\s*,', ',', val)
            val = val.strip().strip(',')
            return val.strip()

        for key in expanded:
            expanded[key] = clean_val(expanded[key])

        # Apply structural expansions in Python based on keywords in the original user prompt
        prompt_lower = prompt.lower()

        # 1. Subject enhancements for mechanical items (symmetry, panel lines, rigid structure)
        machinery_words = ["ship", "spaceship", "vehicle", "satellite", "machine", "robot", "mechanical", "drone", "rover", "cube"]
        if any(w in prompt_lower for w in machinery_words):
            machinery_kw = "perfect geometric symmetry, crisp panel lines, precise engineering blueprint structure, rigid hard-surface panels, straight mechanical lines, zero organic warping"
            if "symmetry" not in expanded["subject"].lower():
                expanded["subject"] = f"{expanded['subject']}, {machinery_kw}"

        # 2. Environment enhancements for cosmic/wormhole items
        cosmic_words = ["wormhole", "portal", "black hole", "galaxy", "nebula", "vortex"]
        if any(w in prompt_lower for w in cosmic_words):
            cosmic_kw = "a swirling gravitational vortex, gravitational lensing bending surrounding light, concentric rings of intense light, accretion disk, deep gravitational funnel structure"
            if "vortex" not in expanded["environment"].lower():
                expanded["environment"] = f"{expanded['environment']}, {cosmic_kw}"

        # 3. Color enhancement (prevent color leakage or overriding by other styling presets)
        color_words = ["red", "blue", "green", "white", "yellow", "orange", "purple", "pink", "black", "gold"]
        for cw in color_words:
            if f" {cw} " in f" {prompt_lower} ":
                color_kw = f"vibrant {cw} coloring, predominantly {cw} accents, highly visible {cw} color scheme"
                if color_kw not in expanded["subject"].lower():
                    expanded["subject"] = f"{expanded['subject']}, {color_kw}"

        # 4. Text/Title preservation (extract any quoted title and reinforce typography instructions)
        quoted_titles = re.findall(r'["\']([^"\']+)["\']', prompt)
        if quoted_titles:
            for title in quoted_titles:
                title_kw = f'bold typography movie title text "{title}", centered poster title layout, clean lettering'
                if title.lower() not in expanded["subject"].lower() and title.lower() not in expanded["style"].lower():
                    expanded["subject"] = f'{expanded["subject"]}, featuring the {title_kw}'

        # 5. Category-specific enhancements
        if category and subcategory:
            try:
                from lumaforge.category_prompts import get_category_prompts
                category_prompt = get_category_prompts(category, subcategory)
                if category_prompt:
                    expanded["style"] = f"{expanded['style']}, {category_prompt}"
            except Exception as e:
                print(f"[OllamaClient Warning] Failed to apply category enhancement: {e}")

        # Consolidate into full prompt
        parts = [
            expanded.get("subject", ""),
            expanded.get("action", ""),
            expanded.get("environment", ""),
            expanded.get("style", ""),
            expanded.get("lighting", ""),
            expanded.get("camera", ""),
            expanded.get("mood", ""),
            expanded.get("quality_emphasis", "")
        ]
        expanded["full_prompt"] = ", ".join([str(p) for p in parts if p])
        return expanded
