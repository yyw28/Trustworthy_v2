#!/usr/bin/env python3
"""
Evaluate best HuBERT checkpoint on test dataset.

Usage:
    python evaluate_checkpoint.py --combined_csv combined_dataset.csv --tis_dir /path/to/tis --cxd_dir /path/to/cxd
"""

import argparse
import os
from pathlib import Path
import torch
import lightning.pytorch as pl

from tspeech.hubert.model.htmodel import HTModel
from tspeech.hubert.data.ht_datamodule import HTDataModule


def find_best_checkpoint(log_dir: str = "lightning_logs/hubert-finetune"):
    """Find best_model.ckpt in the latest version directory."""
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    version_dirs = sorted([d for d in log_path.iterdir() if d.is_dir() and d.name.startswith("version_")])
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found in {log_dir}")
    
    best_model = version_dirs[-1] / "checkpoints" / "best_model.ckpt"
    if not best_model.exists():
        raise FileNotFoundError(f"best_model.ckpt not found in {best_model.parent}")
    
    return str(best_model)


def main():
    parser = argparse.ArgumentParser(description="Evaluate best HuBERT checkpoint on test dataset")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint (auto-finds best_model.ckpt if not provided)")
    parser.add_argument("--combined_csv", type=str, required=True, help="Path to combined dataset CSV")
    parser.add_argument("--tis_dir", type=str, default=None, help="Path to TIS dataset directory")
    parser.add_argument("--cxd_dir", type=str, default=None, help="Path to CXD dataset directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (reduce if OOM: try 1)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "mps"])
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs/devices to use (WARNING: Multi-GPU uses more memory! Use 1 for evaluation)")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"], help="Precision (16-mixed uses less memory)")
    parser.add_argument("--log_dir", type=str, default="lightning_logs/hubert-finetune")
    
    args = parser.parse_args()
    
    # Set memory management environment variable if not set
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory before loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved / {total:.2f} GB total")
        
        # Check for other processes using GPU
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=2)
            if result.stdout.strip():
                print("WARNING: Other processes are using GPU memory:")
                print(result.stdout)
        except:
            pass
    
    # Find checkpoint
    checkpoint_path = args.checkpoint_path or find_best_checkpoint(args.log_dir)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("=" * 60)
    print("HuBERT Test Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # Load model with memory optimizations
    print("Loading model...")
    with torch.no_grad():  # Don't track gradients during loading
        model = HTModel.load_from_checkpoint(checkpoint_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()  # Ensure eval mode (no gradients)
    
    # Move to GPU if needed
    if torch.cuda.is_available() and args.accelerator == "gpu":
        model = model.cuda()
        torch.cuda.empty_cache()
    
    # Clear cache after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory after loading model: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved / {total:.2f} GB total")
        
        if allocated > total * 0.9:
            print("WARNING: GPU memory usage is very high! Consider:")
            print("  1. Reducing batch_size (current: {})".format(args.batch_size))
            print("  2. Using CPU evaluation: --accelerator cpu")
            print("  3. Killing other GPU processes")
    
    # Setup data
    datasets_dict = {}
    if args.tis_dir:
        datasets_dict["tis"] = args.tis_dir
    if args.cxd_dir:
        datasets_dict["cxd"] = args.cxd_dir
    
    data_module = HTDataModule(
        datasets=datasets_dict,
        combined_csv=args.combined_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup("test")
    
    # Force single GPU for evaluation to save memory
    if args.accelerator == "gpu":
        if args.devices > 1:
            print(f"WARNING: Multi-GPU evaluation uses more memory. Forcing devices=1 for memory efficiency.")
        args.devices = 1  # Always use single GPU for evaluation
    
    # Run test with memory optimizations
    trainer_kwargs = {
        "accelerator": args.accelerator,
        "devices": args.devices if args.accelerator != "cpu" else "auto",
        "logger": False,
        "enable_progress_bar": True,
        "precision": args.precision if args.accelerator != "cpu" else "32-true",  # Use mixed precision to save memory
    }
    
    # Only set strategy for multi-GPU (but we force single GPU, so this won't happen)
    # For single GPU, Lightning uses default strategy automatically
    
    trainer = pl.Trainer(**trainer_kwargs)
    results = trainer.test(model, data_module)
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    test_results = results[0]
    print(f"Test Loss: {test_results.get('test_loss', 'N/A'):.6f}")
    print(f"Test F1 Score: {test_results.get('test_f1', 'N/A'):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
