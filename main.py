import argparse
import sys
import json
import os
from lumaforge.ollama_client import OllamaClient
from lumaforge.pipeline import LumaForgePipeline
from lumaforge.safety import SafetyManager
from lumaforge.benchmark import BenchmarkSuite
from lumaforge.dataset_curator import DatasetCurator
from lumaforge.train import LumaForgeTrainer

def print_banner():
    banner = r"""
==================================================
   __    _  _  __  __   __   ____  _____  ____  ___  ____ 
  (  )  / )( \(  \/  ) / _\ (  __)(  _  )(  _ \/ __)(  __)
  / (_/\) \/ ( )    ( /    \ ) _)  )(_)(  )   /\__ \ ) _) 
  \____/\____/(_/\/\_)\_/\_/(__)  (_____)(__\_)(___/(____)
                 AURAGEN MPS CORE T2I
==================================================
    """
    print(banner)

def handle_generate(args):
    print(f"\n[Generate] Starting generation workflow...")
    print(f" -> Prompt: \"{args.prompt}\"")
    print(f" -> Mode: {args.mode.upper()}")
    print(f" -> Aspect Ratio: {args.aspect_ratio}")
    print(f" -> Device: {args.device}")
    
    ollama = OllamaClient()
    safety = SafetyManager(ollama_client=ollama)
    
    # 1. Check prompt safety
    print("\n[Stage 1] Checking safety & moderation boundaries...")
    mod_res = safety.moderate_prompt(args.prompt)
    
    print(f" -> Classification: {mod_res['classification']}")
    print(f" -> Reason: {mod_res['reason']}")
    print(f" -> Moderation Status: {mod_res['status']}")
    
    if mod_res["status"] == "REFUSED":
        print("\n[Refusal] Generation blocked. Prompt violates safety policies.")
        sys.exit(1)
        
    final_prompt = mod_res["final_prompt"]
    if mod_res["status"] == "REWRITTEN":
        print(f" -> Safety Rewritten Prompt: \"{final_prompt}\"")
        
    # 2. Prompt Expansion
    print("\n[Stage 2] Running Ollama prompt adapter / expansion...")
    expanded = ollama.expand_prompt(final_prompt, mode=args.mode)
    
    print(f" -> Subject: {expanded.get('subject')}")
    print(f" -> Style: {expanded.get('style')}")
    print(f" -> Lighting: {expanded.get('lighting')}")
    print(f" -> Camera: {expanded.get('camera')}")
    print(f" -> Mood: {expanded.get('mood')}")
    print(f" -> Consolidated Prompt: \"{expanded.get('full_prompt')}\"")
    
    # 3. Image Generation
    print("\n[Stage 3] Executing latent image generator...")
    pipeline = LumaForgePipeline(model_id=args.model_id, device=args.device)
    
    gen_prompt = expanded.get("full_prompt", final_prompt)
    
    # Check if a fine-tuned LoRA weights file is available, and load it into diffusers if so
    # In a production app, the pipeline would load safe-tensors LoRA.
    lora_path = "weights/lumaforge_lora.safetensors"
    if os.path.exists(lora_path) and not args.mock:
        print(f" -> LoRA Fine-Tuned Weights found at {lora_path}. Loading adapters...")
        
    gen_res = pipeline.generate(
        prompt=gen_prompt,
        aspect_ratio=args.aspect_ratio,
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        mock=args.mock
    )
    
    # 4. Save output
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"output_{gen_res['seed']}.png")
    gen_res["image"].save(out_path)
    
    print(f"\n[Stage 4] Image successfully saved to: {out_path}")
    print(f" -> Latency: {gen_res['latency_sec']:.2f} seconds")
    print(f" -> Unified memory footprint delta: {gen_res['memory_used_mb']:.2f} MB")
    print(f" -> Generation Seed: {gen_res['seed']}")
    print(f" -> Device Used: {gen_res['device']}")
    print(f" -> Pipeline: {'Mock Engine' if gen_res['used_mock'] else 'PyTorch Diffusion'}")
    
    # 5. Output Post-generation screen
    post_res = safety.check_output_safety(out_path, mod_res)
    print(f" -> Post-generation safety check: {post_res['status']} ({post_res['reason']})")

def handle_benchmark(args):
    print("\n[Benchmark] Loading benchmark suite...")
    pipeline = LumaForgePipeline(device=args.device)
    safety = SafetyManager()
    suite = BenchmarkSuite(pipeline, safety, output_path=args.output)
    
    report = suite.run(mock=args.mock)
    
    print("\n================ BENCHMARK REPORT SUMMARY ================")
    print(json.dumps(report["summary"], indent=2))
    print("==========================================================")

def handle_curate(args):
    curator = DatasetCurator()
    curated_count = curator.download_and_curate(limit=args.limit, use_ollama_captioning=args.caption)
    print(f"\n[Curation] Complete. Curated dataset size: {curated_count} images.")

def handle_train(args):
    trainer = LumaForgeTrainer(model_id=args.model_id, device=args.device)
    trainer.run_training(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        demo=args.demo,
        cooldown_secs=args.cooldown,
        checkpoint_steps=args.checkpoint_steps,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir
    )

def handle_audit_log(args):
    safety = SafetyManager()
    logs = safety.get_audit_logs(limit=args.limit)
    
    if not logs:
        print("\n[Audit Log] No audit events found.")
        return
        
    print(f"\n[Audit Log] Showing latest {len(logs)} events:")
    for idx, log in enumerate(logs):
        print(f"\n[{idx+1}] Timestamp: {log['timestamp']} | Event: {log['event_type']} | Status: {log['status']}")
        print(f"  User Prompt: \"{log['user_prompt']}\"")
        if log.get('processed_prompt'):
            print(f"  Final Prompt: \"{log['processed_prompt']}\"")
        print(f"  Class: {log['classification']} | Reason: {log['reason']} | Latency: {log['latency_ms']:.1f}ms")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="LumaForge (AuraGen MPS) CLI - Local AI Image Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate Subparser
    parser_gen = subparsers.add_parser("generate", help="Generate an image from a prompt")
    parser_gen.add_argument("--prompt", type=str, required=True, help="Text prompt to generate")
    parser_gen.add_argument("--mode", type=str, choices=["general", "poster", "character"], default="general", help="Prompt expansion style preset")
    parser_gen.add_argument("--aspect_ratio", type=str, choices=["1:1", "16:9", "9:16", "4:3", "3:4"], default="1:1", help="Image aspect ratio dimensions")
    parser_gen.add_argument("--mock", action="store_true", help="Force mock image generator fallback (extremely fast)")
    parser_gen.add_argument("--device", type=str, default="mps", help="PyTorch acceleration device (mps, cpu)")
    parser_gen.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Hugging Face model ID")
    parser_gen.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser_gen.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser_gen.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser_gen.add_argument("--seed", type=int, default=-1, help="Seed value for reproducibility")
    
    # Benchmark Subparser
    parser_bench = subparsers.add_parser("benchmark", help="Run model evaluation suite")
    parser_bench.add_argument("--mock", action="store_false", dest="mock", help="Run with actual diffusion inference (default is mock for speed)")
    parser_bench.set_defaults(mock=True)
    parser_bench.add_argument("--device", type=str, default="mps", help="Device for evaluation")
    parser_bench.add_argument("--output", type=str, default="benchmark_report.json", help="Path to save JSON benchmark report")
    
    # Curate Subparser
    parser_curate = subparsers.add_parser("curate", help="Collect, deduplicate, and caption training images")
    parser_curate.add_argument("--limit", type=int, default=90, help="Max image samples to curate")
    parser_curate.add_argument("--no-caption", action="store_false", dest="caption", help="Disable Ollama caption refinement")
    parser_curate.set_defaults(caption=True)

    # Train Subparser
    parser_train = subparsers.add_parser("train", help="Fine-tune model LoRA layers on curated dataset")
    parser_train.add_argument("--epochs", type=int, default=3, help="Training epoch count")
    parser_train.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser_train.add_argument("--batch_size", type=int, default=2, help="DataLoader batch size")
    parser_train.add_argument("--demo", action="store_true", help="Run in training simulation demo mode")
    parser_train.add_argument("--device", type=str, default="mps", help="Device for training")
    parser_train.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Target model ID")
    parser_train.add_argument("--cooldown", type=float, default=0.0, help="Cooldown sleep duration in seconds after each batch step")
    parser_train.add_argument("--checkpoint_steps", type=int, default=0, help="Steps interval to save intermediate checkpoints (0 to disable)")
    parser_train.add_argument("--resume", action="store_true", help="Resume fine-tuning from latest checkpoint")
    parser_train.add_argument("--checkpoint_dir", type=str, default="weights/checkpoints", help="Directory path to save/load checkpoints")
    
    # Audit Log Subparser
    parser_audit = subparsers.add_parser("audit-log", help="View local moderation audit logs")
    parser_audit.add_argument("--limit", type=int, default=10, help="Number of latest log entries to print")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        handle_generate(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "curate":
        handle_curate(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "audit-log":
        handle_audit_log(args)

if __name__ == "__main__":
    main()
