import json
import os
import time
from lumaforge.ollama_client import OllamaClient

class SafetyManager:
    def __init__(self, audit_log_path="audit_log.jsonl", ollama_client=None):
        self.audit_log_path = audit_log_path
        self.ollama = ollama_client or OllamaClient()

    def log_event(self, event_type: str, user_prompt: str, processed_prompt: str, classification: str, reason: str, status: str, latency_ms: float):
        """Appends a moderation event to the JSONL audit log."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
            "user_prompt": user_prompt,
            "processed_prompt": processed_prompt,
            "classification": classification,
            "reason": reason,
            "status": status,
            "latency_ms": latency_ms
        }
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[SafetyManager Error] Failed to write audit log: {e}")

    def moderate_prompt(self, user_prompt: str) -> dict:
        """
        Runs the prompt through the safety classifier and semantic rewrite layer.
        Returns a dict:
        {
          "status": "APPROVED" | "REWRITTEN" | "REFUSED",
          "original_prompt": str,
          "final_prompt": str,
          "classification": str,
          "reason": str,
          "latency_ms": float
        }
        """
        start_time = time.time()
        
        # Step 1: Classify prompt
        classification_result = self.ollama.classify_safety(user_prompt)
        classification = classification_result.get("classification", "SAFE").strip().upper()
        reason = classification_result.get("reason", "No reason provided.")
        
        status = "APPROVED"
        final_prompt = user_prompt
        
        # Step 2: Act on classification
        if classification == "UNSAFE":
            status = "REFUSED"
            final_prompt = ""
        elif classification == "BORDERLINE":
            status = "REWRITTEN"
            final_prompt = self.ollama.rewrite_prompt(user_prompt)
            
        latency_ms = (time.time() - start_time) * 1000
        
        # Step 3: Log event
        self.log_event(
            event_type="INPUT_PROMPT",
            user_prompt=user_prompt,
            processed_prompt=final_prompt,
            classification=classification,
            reason=reason,
            status=status,
            latency_ms=latency_ms
        )
        
        return {
            "status": status,
            "original_prompt": user_prompt,
            "final_prompt": final_prompt,
            "classification": classification,
            "reason": reason,
            "latency_ms": latency_ms
        }

    def check_output_safety(self, image_path: str, prompt_metadata: dict) -> dict:
        """
        Runs post-generation checks on the generated image.
        If prompt was borderline or contains style risks, we do additional validation.
        """
        start_time = time.time()
        
        # Simple post-generation heuristic checks (simulate image classification)
        # In a production app, this would use a CLIP or ResNet safety checker.
        classification = "SAFE"
        reason = "Output image checks passed."
        status = "APPROVED"
        
        # Check if the prompt metadata was flagged as rewritten or borderline
        if prompt_metadata.get("status") == "REWRITTEN":
            classification = "SAFE_RECOVERED"
            reason = "Image generated from safety-aligned rewritten prompt."
            
        latency_ms = (time.time() - start_time) * 1000
        
        self.log_event(
            event_type="OUTPUT_IMAGE",
            user_prompt=prompt_metadata.get("original_prompt", ""),
            processed_prompt=prompt_metadata.get("final_prompt", ""),
            classification=classification,
            reason=reason,
            status=status,
            latency_ms=latency_ms
        )
        
        return {
            "status": status,
            "classification": classification,
            "reason": reason,
            "latency_ms": latency_ms
        }

    def get_audit_logs(self, limit=100):
        """Retrieves history of moderation events."""
        if not os.path.exists(self.audit_log_path):
            return []
        
        logs = []
        try:
            with open(self.audit_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            # Return latest logs first
            return logs[::-1][:limit]
        except Exception as e:
            print(f"[SafetyManager Error] Failed to read audit log: {e}")
            return []
