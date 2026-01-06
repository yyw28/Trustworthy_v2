from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

# HuBERT implementation
from tspeech.hubert.model.htmodel import HTModel
from tspeech.hubert.data.ht_datamodule import HTDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(HTModel, HTDataModule)


def main():
    """Main function to run HuBERT fine-tuning."""
    torch.set_float32_matmul_precision("high")
    
    # Initialize data module
    data_module = HTDataModule(
        datasets={
            "tis": "/Users/yuwenyu/Desktop/trustworthy_classifier_standalone/tis",
            "cxd": "/Users/yuwenyu/Desktop/trustworthy_matt/Trustworthy_v2/CXD"
        },
        combined_csv="/Users/yuwenyu/Desktop/trustworthy_matt/Trustworthy_v2/combined_dataset.csv",
        batch_size=4,
        num_workers=2,
    )
    
    # Initialize model
    model = HTModel(
        hubert_model_name="facebook/hubert-base-ls960",
        trainable_layers=2,
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            mode="min",
            monitor="validation_loss",
            patience=5,
            min_delta=0.001,
        ),
        ModelCheckpoint(
            filename="best-checkpoint-{epoch}-{validation_loss:.5f}-{validation_accuracy:.5f}",
            mode="min",
            monitor="validation_loss",
            save_top_k=1,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(save_last=True),
    ]
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="hubert-finetune",
        default_hp_metric=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=100,
        accumulate_grad_batches=2,
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed",
    )
    
    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    import sys
    
    # If config file provided, use CLI; otherwise use main function
    if len(sys.argv) > 1 and sys.argv[1] == "fit" and "--config" in sys.argv:
        cli_main()
    else:
        main()

