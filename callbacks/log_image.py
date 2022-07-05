from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class LogImagesCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def _log_images(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        prefix: str,
    ) -> None:
        """Log 5 reconstructed images & 5 generated images by sampling latent vectors."""
        original = outputs.get("input")[0:5]
        recons = outputs.get("reconstructed")[0:5]
        with torch.no_grad():
            pl_module.eval()
            gen = pl_module.generate_img(5)
            pl_module.train()

        trainer.logger.log_image(key=f"{prefix}_original", images=[original])
        trainer.logger.log_image(key=f"{prefix}_reconstructed", images=[recons])
        trainer.logger.log_image(key=f"{prefix}_generated", images=[gen])

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """log images after first validation batch"""
        if batch_idx == 0:
            self._log_images(trainer, pl_module, outputs, prefix="val")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """log images after first validation batch"""
        if batch_idx == 0:
            self._log_images(trainer, pl_module, outputs, prefix="test")
