#!/usr/bin/env python3
"""
Train TTS model with RL using TIS dataset.

This script:
1. Converts TIS data to TTS format (if needed)
2. Trains TTS model with RL using TIS audio files

Usage:
    python run_tts_rl_with_tis.py \
        --tis_dir /path/to/tis \
        --hubert_checkpoint /path/to/hubert/checkpoint.ckpt \
        [--use_asr] \
        [--placeholder_text "Your text here"]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train TTS with RL using TIS data")
    
    # TIS data arguments
    parser.add_argument("--tis_dir", type=str, required=True, help="Path to TIS dataset directory")
    parser.add_argument("--use_asr", action="store_true", help="Use Whisper ASR to extract transcripts")
    parser.add_argument("--placeholder_text", type=str, default="This is a trustworthy statement.", help="Placeholder text if not using ASR")
    
    # HuBERT checkpoint
    parser.add_argument("--hubert_checkpoint", type=str, required=True, help="Path to trained HuBERT checkpoint")
    
    # TTS training arguments (passed to run_tts_with_rl.py)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--rl_temperature", type=float, default=1.0, help="RL sampling temperature")
    parser.add_argument("--rl_entropy_coef", type=float, default=0.01, help="RL entropy coefficient")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "mps"])
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    
    # Output directory for TTS CSV files
    parser.add_argument("--tts_csv_dir", type=str, default="./tis_tts_data", help="Directory for TTS CSV files")
    
    args = parser.parse_args()
    
    tis_dir = Path(args.tis_dir)
    tts_csv_dir = Path(args.tts_csv_dir)
    
    # Step 1: Check if TTS CSV files exist, if not create them
    train_csv = tts_csv_dir / "train.csv"
    val_csv = tts_csv_dir / "val.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        print("=" * 80)
        print("Step 1: Converting TIS data to TTS format")
        print("=" * 80)
        
        prepare_script = Path(__file__).parent / "prepare_tis_for_tts.py"
        if not prepare_script.exists():
            raise FileNotFoundError(f"prepare_tis_for_tts.py not found: {prepare_script}")
        
        cmd = [
            sys.executable,
            str(prepare_script),
            "--tis_dir", str(tis_dir),
            "--output_dir", str(tts_csv_dir),
        ]
        
        if args.use_asr:
            cmd.append("--use_asr")
        
        if args.placeholder_text:
            cmd.extend(["--placeholder_text", args.placeholder_text])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        if not train_csv.exists() or not val_csv.exists():
            raise RuntimeError(f"Failed to create TTS CSV files in {tts_csv_dir}")
        
        print("✓ TTS CSV files created")
    else:
        print(f"✓ Using existing TTS CSV files in {tts_csv_dir}")
    
    # Step 2: Train TTS with RL
    print("\n" + "=" * 80)
    print("Step 2: Training TTS model with RL")
    print("=" * 80)
    
    run_tts_script = Path(__file__).parent / "run_tts_with_rl.py"
    if not run_tts_script.exists():
        raise FileNotFoundError(f"run_tts_with_rl.py not found: {run_tts_script}")
    
    cmd = [
        sys.executable,
        str(run_tts_script),
        "--tts_data_dir", str(tis_dir.absolute()),
        "--train_csv", str(train_csv.absolute()),
        "--val_csv", str(val_csv.absolute()),
        "--hubert_checkpoint", args.hubert_checkpoint if args.hubert_checkpoint else "",
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--max_epochs", str(args.max_epochs),
        "--rl_temperature", str(args.rl_temperature),
        "--rl_entropy_coef", str(args.rl_entropy_coef),
        "--learning_rate", str(args.learning_rate),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("\n" + "=" * 80)
    
    result = subprocess.run(cmd, check=True)
    
    print("\n" + "=" * 80)
    print("TTS training with RL complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
