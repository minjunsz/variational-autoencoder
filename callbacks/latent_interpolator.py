from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


class LatentInterpolator(Callback):
    """interpolate first two latent dimension and log the generated images"""

    def __init__(self, interpolate_epoch_interval: int) -> None:
        super().__init__()
