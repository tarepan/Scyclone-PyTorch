from typing import Tuple
import itertools

import torch
from torch.nn import functional as F
from torch.tensor import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

from torchaudio.transforms import GriffinLim

from .modules import Generator, Discriminator

# ## Glossary
# - G: Generator
# - D: Discriminator
# - X2Y: map X to Y (e.g. B2A)


class Scyclone(pl.LightningModule):
    """Scyclone, non-parallel voice conversion model.
    """

    def __init__(self, sampling_rate: int, noiseless_D: bool = False):

        super().__init__()

        # params
        self.hparams = {
            # "λcy and λid were set to 10 and 1 in Eq. 1" in Scyclone paper
            # self.weight_adv = 1 # standard
            "weight_cycle": 10,
            "weight_identity": 1,
            # "m is a parameter of the hinge loss and is set to (...) 0.5 in our experiments" in Scyclone paper
            "hinge_offset_D": 0.5,
            "learning_rate": 2.0 * 1e-4,
            "sampling_rate": sampling_rate
        }
        self.save_hyperparameters()

        self.G_A2B = Generator()
        self.G_B2A = Generator()
        self.D_A = Discriminator(noise_sigma=0 if noiseless_D else 0.01)
        self.D_B = Discriminator(noise_sigma=0 if noiseless_D else 0.01)

        self.griffinLim = GriffinLim(n_fft=254, n_iter=256)

    def forward(self, x: Tensor) -> Tensor:

        return self.G_A2B(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int):
        """Min-Max adversarial training (G:D = 1:1)
        """

        real_A, real_B = batch

        # Generator training
        if optimizer_idx == 0:
            # Generator adversarial losses: hinge loss (from Scyclone paper eq.1)
            fake_B = self.G_A2B(real_A)
            pred_fake_B = self.D_B(torch.narrow(fake_B, 2, 16, 128))
            loss_adv_G_A2B = torch.mean(F.relu(-1.0 * pred_fake_B))
            fake_A = self.G_B2A(real_B)
            pred_fake_A = self.D_A(torch.narrow(fake_A, 2, 16, 128))
            loss_adv_G_B2A = torch.mean(F.relu(-1.0 * pred_fake_A))

            # cycle consistency losses: L1 loss (from Scyclone paper eq.1)
            cycled_A = self.G_B2A(fake_B)
            loss_cycle_ABA = F.l1_loss(cycled_A, real_A)
            cycled_B = self.G_A2B(fake_A)
            loss_cycle_BAB = F.l1_loss(cycled_B, real_B)

            # identity mapping losses: L1 loss (from Scyclone paper eq.1)
            same_B = self.G_A2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B)
            same_A = self.G_B2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A)

            # Total loss
            loss_G = (
                loss_adv_G_A2B
                + loss_adv_G_B2A
                + loss_cycle_ABA * self.hparams["weight_cycle"]
                + loss_cycle_BAB * self.hparams["weight_cycle"]
                + loss_identity_A * self.hparams["weight_identity"]
                + loss_identity_B * self.hparams["weight_identity"]
            )

            log = {
                "Loss/G_total": loss_G,
                "Loss/Adv/G_B2A": loss_adv_G_B2A,
                "Loss/Adv/G_A2B": loss_adv_G_A2B,
                "Loss/Cyc/A2B2A": loss_cycle_ABA * self.hparams["weight_cycle"],
                "Loss/Cyc/B2A2B": loss_cycle_BAB * self.hparams["weight_cycle"],
                "Loss/Id/A2A": loss_identity_A * self.hparams["weight_identity"],
                "Loss/Id/B2B": loss_identity_B * self.hparams["weight_identity"],
            }
            out = {"loss": loss_G, "log_losses": log}

        # Discriminator training
        elif optimizer_idx == 1:
            m = self.hparams["hinge_offset_D"]

            # Adversarial loss: hinge loss (from Scyclone paper eq.1)
            # D_A
            ## Real loss
            ### edge cut: [B, C, L] => [B, C, L_cut]
            pred_A_real = self.D_A(torch.narrow(real_A, 2, 16, 128))
            loss_D_A_real = torch.mean(F.relu(m - pred_A_real))
            ## Fake loss
            fake_A = self.G_B2A(real_B)  # no_grad by PyTorch-Lightning
            pred_A_fake = self.D_A(torch.narrow(fake_A, 2, 16, 128))
            loss_D_A_fake = torch.mean(F.relu(m + pred_A_fake))
            ## D_A total loss
            loss_D_A = loss_D_A_real + loss_D_A_fake

            # D_B
            ## Real loss
            pred_B_real = self.D_B(torch.narrow(real_B, 2, 16, 128))
            loss_D_B_real = torch.mean(F.relu(m - pred_B_real))
            ## Fake loss
            fake_B = self.G_A2B(real_A)  # no_grad by PyTorch-Lightning
            pred_B_fake = self.D_B(torch.narrow(fake_B, 2, 16, 128))
            loss_D_B_fake = torch.mean(F.relu(m + pred_B_fake))
            ## D_B total loss
            loss_D_B = loss_D_B_real + loss_D_B_fake

            # Total
            loss_D = loss_D_A + loss_D_B

            log = {
                "Loss/D_total": loss_D,
                "Loss/D_A": loss_D_A,
                "Loss/D_B": loss_D_B,
            }
            out = {"loss": loss_D, "log_losses": log}

        else:
            raise ValueError(f"invarid optimizer_idx: {optimizer_idx}")
        return out

    def training_step_end(self, out):

        # logging
        for name, value in out["log_losses"].items():
            self.log(name, value, on_step=False, on_epoch=True)
        return out

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):

        # loss calculation
        o_G = self.training_step(batch, batch_idx, 0)
        o_D = self.training_step(batch, batch_idx, 1)
        # sample conversion
        real_A, real_B = batch
        fake_B = self.G_A2B(real_A)
        fake_A = self.G_B2A(real_B)
        # ReLU for preventing minus value for GriffinLim
        fake_B_wave = self.griffinLim(F.relu(fake_B))
        fake_A_wave = self.griffinLim(F.relu(fake_A))

        self.log("val_loss", o_D["loss"])
        return {
            "val_loss": o_D["loss"],
            "wave": {"Validation/A2B": fake_B_wave, "Validation/B2A": fake_A_wave},
            "loss": {"G": o_G["log_losses"], "D": o_D["log_losses"]},
        }

    def validation_step_end(self, out) -> None:
        
        # waveform logging
        for i in range(0, 2):
            a2b = out["wave"]["Validation/A2B"][i]
            max_a2b = torch.max(a2b, 0, keepdim=True).values
            min_a2b = torch.min(a2b, 0, keepdim=True).values
            scaler_a2b = torch.max(torch.cat((torch.abs(max_a2b), torch.abs(min_a2b))))
            self.logger.experiment.add_audio(
                "Validation/A2B",
                torch.reshape(a2b / scaler_a2b, (1, -1)),
                global_step=self.current_epoch,
                sample_rate=self.hparams["sampling_rate"],
            )
            b2a = out["wave"]["Validation/B2A"][i]
            max_b2a = torch.max(b2a, 0, keepdim=True).values
            min_b2a = torch.min(b2a, 0, keepdim=True).values
            scaler_b2a = torch.max(torch.cat((torch.abs(max_b2a), torch.abs(min_b2a))))
            self.logger.experiment.add_audio(
                "Validation/B2A",
                torch.reshape(b2a / scaler_b2a, (1, -1)),
                global_step=self.current_epoch,
                sample_rate=self.hparams["sampling_rate"],
            )
        # loss logging
        for gd in ["G", "D"]:
            for name, value in out["loss"][gd].items():
                self.log(f"Validation/{name}", value, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        pass

    def configure_optimizers(self):
        """Set up G/D optimizers
        """

        decay_rate = 0.1
        decay_iter = 100000
        optim_G = Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=self.hparams["learning_rate"],
            betas=(0.5, 0.999),
        )
        sched_G = {
            "scheduler": StepLR(optim_G, decay_iter, decay_rate),
            "interval": "step",
        }
        optim_D = Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.hparams["learning_rate"],
            betas=(0.5, 0.999),
        )
        sched_D = {
            "scheduler": StepLR(optim_D, decay_iter, decay_rate),
            "interval": "step",
        }
        return [optim_G, optim_D], [sched_G, sched_D]
