#!/usr/bin/env python3
"""
Simple script to run HubERT training on server.

Usage:
    # If CSV has relative paths (most common):
    python run_hubert_training.py --combined_csv /path/to/combined.csv --tis_dir /path/to/tis --cxd_dir /path/to/cxd
"""

import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Train HubERT classifier")
    parser.add_argument("--combined_csv", type=str, required=True, help="Path to combined dataset CSV")
    parser.add_argument("--tis_dir", type=str, default=None, help="Path to TIS dataset directory (required if CSV has relative paths)")
    parser.add_argument("--cxd_dir", type=str, default=None, help="Path to CXD dataset directory (required if CSV has relative paths)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "mps"], help="Accelerator type")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs/devices to use (e.g., 2 for 2 GPUs)")
    parser.add_argument("--strategy", type=str, default=None, help="Training strategy (auto-detected for multi-GPU: 'ddp' or 'ddp_find_unused_parameters_true')")
    parser.add_argument("--trainable_layers", type=int, default=2, help="Number of trainable HubERT layers")
    parser.add_argument(
        "--hubert_model",
        type=str,
        default="facebook/hubert-xlarge-ls960-ft",
        help="HuBERT model name. Options: "
             "facebook/hubert-base-ls960 (base, fastest), "
             "facebook/hubert-large-ls960-ft (large, better performance), "
             "facebook/hubert-xlarge-ls960-ft (xlarge, best performance, requires more memory)"
    )
    
    args = parser.parse_args()
    
    # Set precision for matrix operations
    torch.set_float32_matmul_precision("high")
    
    print("=" * 60)
    print("HubERT Classifier Training")
    print("=" * 60)
    print(f"Combined CSV: {args.combined_csv}")
    if args.tis_dir:
        print(f"TIS Directory: {args.tis_dir}")
    if args.cxd_dir:
        print(f"CXD Directory: {args.cxd_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Devices: {args.devices}")
    print(f"Trainable Layers: {args.trainable_layers}")
    print(f"HuBERT Model: {args.hubert_model}")
    print("=" * 60)
    
    # Import here to avoid issues
    from tspeech.hubert.model.htmodel import HTModel
    from tspeech.hubert.data.ht_datamodule import HTDataModule
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
    
    # Initialize data module
    print("\n[1/3] Initializing data module...")
    
    # Build datasets dict from provided arguments
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
    
    # Initialize model
    print("[2/3] Initializing HubERT model...")
    model = HTModel(
        hubert_model_name=args.hubert_model,
        trainable_layers=args.trainable_layers,
    )
    
    # Setup logger
    print("[3/3] Setting up logger and callbacks...")
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="hubert-finetune",
    )
    
    # Custom callback to print train and eval loss after each epoch
    class PrintMetricsCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            train_loss = metrics.get("training_loss", None)
            if train_loss is not None:
                print(f"\nEpoch {trainer.current_epoch}: Train Loss = {train_loss:.6f}")
        
        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            val_loss = metrics.get("validation_loss", None)
            val_acc = metrics.get("validation_accuracy", None)
            val_f1 = metrics.get("validation_f1", None)
            
            if val_loss is not None:
                print(f"Epoch {trainer.current_epoch}: Eval Loss = {val_loss:.6f}", end="")
                if val_acc is not None:
                    print(f", Eval Accuracy = {val_acc:.6f}", end="")
                if val_f1 is not None:
                    print(f", Eval F1 = {val_f1:.6f}", end="")
                print()
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=5,
        min_delta=0.001,
    )
    
    callbacks = [
        PrintMetricsCallback(),
        early_stopping,
        ModelCheckpoint(
            filename="best_model",
            mode="min",
            monitor="validation_loss",
            save_top_k=1,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            save_last=True,
        ),
    ]
    
    # Check CUDA availability if using GPU
    if args.accelerator == "gpu":
        if not torch.cuda.is_available():
            print("WARNING: GPU requested but CUDA not available. Switching to CPU.")
            args.accelerator = "cpu"
        else:
            try:
                # Test CUDA with a simple operation
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                num_gpus = torch.cuda.device_count()
                print(f"CUDA available: {num_gpus} GPU(s) detected")
                for i in range(num_gpus):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                if args.devices > num_gpus:
                    print(f"WARNING: Requested {args.devices} devices but only {num_gpus} available. Using {num_gpus}.")
                    args.devices = num_gpus
            except Exception as e:
                print(f"WARNING: CUDA error detected: {e}")
                print("Switching to CPU. If you need GPU, check PyTorch CUDA compatibility.")
                args.accelerator = "cpu"
    
    # Auto-detect strategy for multi-GPU
    strategy = args.strategy
    if strategy is None and args.accelerator == "gpu" and args.devices > 1:
        # Use DDP for multi-GPU training
        strategy = "ddp_find_unused_parameters_true"
        print(f"Using multi-GPU strategy: {strategy}")
    
    # Initialize trainer
    trainer_kwargs = {
        "accelerator": args.accelerator,
        "devices": args.devices if args.accelerator != "cpu" else "auto",
        "precision": "16-mixed" if args.accelerator != "cpu" else "32-true",  # CPU doesn't support 16-mixed
        "max_epochs": args.max_epochs,
        "accumulate_grad_batches": 2,
        "logger": logger,
        "callbacks": callbacks,
    }
    # Only add strategy if it's not None
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(model, data_module)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Check if early stopping was triggered
    if early_stopping.stopped_epoch > 0:
        print("\nEarly Stopping was triggered!")
        print(f"  Stopped at epoch: {early_stopping.stopped_epoch}")
        print(f"  Best validation loss: {early_stopping.best_score:.6f}")
        print(f"  Patience: {early_stopping.patience} epochs without improvement")
    else:
        print("\nTraining completed all epochs (early stopping not triggered)")
    
    print(f"\nCheckpoints saved to: {logger.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

