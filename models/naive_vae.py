# %%
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .types import TorchDevice, TorchTensor


# %%
def encoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(),
    )


def decoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(),
    )


def reparametrization_trick(mu: TorchTensor, logSigma: TorchTensor) -> TorchTensor:
    epsilon = torch.randn_like(logSigma)
    sigma = torch.exp(logSigma)
    return mu + sigma * epsilon


@dataclass
class VAEOutput:
    reconstructed: torch.Tensor
    mu: torch.Tensor
    logSigma: torch.Tensor


class NaiveVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: List[int],
        latent_dim: int,
        lr: float = 5e-4,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 1, 32, 32)

        self.latent_dim = latent_dim
        self.lr = lr
        self.last_hidden_dim = hidden_dim[-1]

        hidden_dim_copy = [*hidden_dim]
        self.encode_blocks = nn.Sequential(
            encoder_block(in_channels, hidden_dim[0]),
            *[
                encoder_block(in_ch, out_ch)
                for in_ch, out_ch in zip(hidden_dim_copy, hidden_dim_copy[1:])
            ],
        )

        hidden_dim_copy.reverse()
        self.decode_blocks = nn.Sequential(
            *[
                decoder_block(in_ch, out_ch)
                for in_ch, out_ch in zip(hidden_dim_copy, hidden_dim_copy[1:])
            ],
        )

        self.fc_mu = nn.Linear(hidden_dim[-1] * 4, latent_dim)
        self.fc_logSigma = nn.Linear(hidden_dim[-1] * 4, latent_dim)
        self.decoder_init = nn.Linear(latent_dim, hidden_dim[-1] * 4)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim[0],
                hidden_dim[0],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dim[0], out_channels=in_channels, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden: torch.Tensor = self.encode_blocks(input)
        hidden = hidden.flatten(start_dim=1)
        mu = self.fc_mu(hidden)
        logSigma = self.fc_logSigma(hidden)

        return (mu, logSigma)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        output = self.decoder_init(latent)
        output = output.view(-1, self.last_hidden_dim, 2, 2)
        output = self.decode_blocks(output)
        output = self.final_layer(output)

        return output

    def forward(self, input: torch.Tensor) -> VAEOutput:
        mu, logSigma = self.encode(input)
        latent = reparametrization_trick(mu, logSigma)
        reconstructed = self.decode(latent)
        return VAEOutput(reconstructed=reconstructed, mu=mu, logSigma=logSigma)

    def loss_function(
        self, original: TorchTensor, output: VAEOutput
    ) -> Tuple[TorchTensor, TorchTensor]:
        recon_loss = F.mse_loss(output.reconstructed, original)
        kld_loss = (
            (
                -0.5
                - output.logSigma
                + 0.5 * ((2 * output.logSigma).exp() + output.mu**2)
            )
            .sum(dim=1)
            .mean(dim=0)
        )
        return recon_loss, kld_loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)
        recon_loss, kld_loss = self.loss_function(x, output)
        scheduler = self.lr_schedulers()

        log_values = {
            "train_total_loss": recon_loss.detach() + kld_loss.detach(),
            "train_reconstruction_loss": recon_loss.detach(),
            "train_KL_divergence": kld_loss.detach(),
        }
        self.log_dict(log_values)
        scheduler.step(recon_loss.detach() + kld_loss.detach())

        return {"loss": recon_loss + kld_loss}

    def validation_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        # this is the test loop
        x, y = batch
        output = self.forward(x)
        recon_loss, kld_loss = self.loss_function(x, output)

        if batch_idx == 0:
            tensorboard = self.logger.experiment
            original = x[0]
            recons = output.reconstructed[0]
            tensorboard.add_image(
                "val_original", original, global_step=self.current_epoch
            )
            tensorboard.add_image(
                "val_reconstructed", recons, global_step=self.current_epoch
            )

        log_values = {
            "val_total_loss": recon_loss.detach() + kld_loss.detach(),
            "val_reconstruction_loss": recon_loss.detach(),
            "val_KL_divergence": kld_loss.detach(),
        }
        self.log_dict(log_values)

    def test_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        # this is the test loop
        x, y = batch
        output = self.forward(x)
        recon_loss, kld_loss = self.loss_function(x, output)

        if batch_idx == 0:
            tensorboard = self.logger.experiment
            original = x[0]
            recons = output.reconstructed[0]
            tensorboard.add_image(
                "test_original", original, global_step=self.current_epoch
            )
            tensorboard.add_image(
                "test_reconstructed", recons, global_step=self.current_epoch
            )

        log_values = {
            "test_total_loss": recon_loss.detach() + kld_loss.detach(),
            "test_reconstruction_loss": recon_loss.detach(),
            "test_KL_divergence": kld_loss.detach(),
        }
        self.log_dict(log_values)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, "min"),
                "monitor": "train_total_loss",
            },
        }

    def reconstruct_img(self, original: torch.Tensor) -> torch.Tensor:
        """
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """
        return self.forward(original).reconstructed

    def generate_img(self, num_imgs: int, device: torch.device) -> torch.Tensor:
        latent = torch.randn(num_imgs, self.latent_dim)
        latent.to(device=device)
        return self.decode(latent)


# %%
