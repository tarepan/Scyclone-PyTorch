from typing import Tuple
import itertools
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.nn import functional as F
from torch.tensor import Tensor

# currently there is no stub in npvcc2016
from torchaudio.transforms import GriffinLim  # type: ignore

from .datamodule import NonParallelSpecDataModule
from .modules import Generator, Discriminator

# codes are inspired by CycleGAN family with PyTorch Lightning https://github.com/HasnainRaz/Fast-AgingGAN/blob/master/gan_module.py

"""
G: Generator
D: Discriminator
A2B: map A to B (c.f. B2A)
"""


class Scyclone(pl.LightningModule):
    """
    Scyclone, non-parallel voice conversion model.
    Origin: Masaya Tanaka, et al.. (2020). Scyclone: High-Quality and Parallel-Data-Free Voice Conversion Using Spectrogram and Cycle-Consistent Adversarial Networks. Arxiv 2005.03334.
    """

    def __init__(self, _=True):
        super().__init__()

        # params
        self.hparams = {
            ## "λcy and λid were set to 10 and 1 in Eq. 1" in Scyclone paper
            ## self.weight_adv = 1 ## standard
            "weight_cycle": 10,
            "weight_identity": 1,
            ## "m is a parameter of the hinge loss and is set to (...) 0.5 in our experiments" in Scyclone paper
            "hinge_offset_D": 0.5,
            "learning_rate": 2.0 * 1e-4,
        }
        self.save_hyperparameters()

        self.G_A2B = Generator()
        self.G_B2A = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()

    def forward(self, x: Tensor) -> Tensor:
        return self.G_A2B(x)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int
    ):
        """
        Min-Max adversarial training (G:D = 1:1)
        """
        real_A, real_B = batch

        # Generator training
        if optimizer_idx == 0:
            # Generator adversarial losses: hinge loss (from Scyclone paper eq.1)
            fake_B = self.G_A2B(real_A)
            pred_fake_B = self.D_B(fake_B)
            loss_GAN_A2B = torch.mean(F.relu(-1.0 * pred_fake_B))
            fake_A = self.G_B2A(real_B)
            pred_fake_A = self.D_A(fake_A)
            loss_GAN_B2A = torch.mean(F.relu(-1.0 * pred_fake_A))

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
                loss_GAN_A2B
                + loss_GAN_B2A
                + loss_cycle_ABA * self.hparams["weight_cycle"]
                + loss_cycle_BAB * self.hparams["weight_cycle"]
                + loss_identity_A * self.hparams["weight_identity"]
                + loss_identity_B * self.hparams["weight_identity"]
            )

            output = {
                "loss": loss_G,
                "log": {
                    "Loss/G_total": loss_G,
                    "Loss/Adv/G_A2B": loss_GAN_A2B,
                    "Loss/Adv/G_B2A": loss_GAN_B2A,
                    "Loss/Cyc/A2B2A": loss_cycle_ABA * self.hparams["weight_cycle"],
                    "Loss/Cyc/B2A2B": loss_cycle_BAB * self.hparams["weight_cycle"],
                    "Loss/Id/A2A": loss_identity_A * self.hparams["weight_identity"],
                    "Loss/Id/B2B": loss_identity_B * self.hparams["weight_identity"],
                },
            }
            ## registration for Discriminator loop
            self.fake_B = fake_B
            self.fake_A = fake_A
            ## Is this needed...?
            self.real_B = real_B
            self.real_A = real_A

            return output

        # Discriminator training
        if optimizer_idx == 1:
            m = self.hparams["hinge_offset_D"]

            # Adversarial loss: hinge loss (from Scyclone paper eq.1)
            # D_A
            ## Real loss
            ### edge cut: [B, C, L] => [B, C, L_cut]
            pred_A_real = self.D_A(torch.narrow(real_A, 2, 16, 128))
            loss_D_A_real = torch.mean(F.relu(m - pred_A_real))
            ## Fake loss
            pred_A_fake = self.D_A(torch.narrow(self.fake_A.detach(), 2, 16, 128))
            loss_D_A_fake = torch.mean(F.relu(m + pred_A_fake))
            ## D_A total loss
            loss_D_A = loss_D_A_real + loss_D_A_fake

            # D_B
            ## Real loss
            pred_B_real = self.D_B(torch.narrow(real_B, 2, 16, 128))
            loss_D_B_real = torch.mean(F.relu(m - pred_B_real))
            ## Fake loss
            pred_B_fake = self.D_B(torch.narrow(self.fake_B.detach(), 2, 16, 128))
            loss_D_B_fake = torch.mean(F.relu(m + pred_B_fake))
            ## D_B total loss
            loss_D_B = loss_D_B_real + loss_D_B_fake

            # Total
            loss_D = loss_D_A + loss_D_B
            output = {
                "loss": loss_D,
                "log": {
                    "Loss/D_total": loss_D,
                    "Loss/D_A": loss_D_A,
                    "Loss/D_B": loss_D_B,
                },
            }
            return output

    def training_step_end(self, out):
        self.log_dict(out["log"], on_step=False, on_epoch=True)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        real_A, real_B = batch
        fake_B = self.G_A2B(real_A)
        fake_A = self.G_B2A(real_B)
        gl = GriffinLim(n_fft=254, n_iter=128)
        fake_B_wave = gl(fake_B)
        fake_A_wave = gl(fake_A)
        return {"log": {"Validation/A2B": fake_B_wave, "Validation/B2A": fake_A_wave}}

    def validation_step_end(self, out) -> None:
        ## self.logger.experiment.xxx (add wave的な)
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        pass

    def configure_optimizers(self):
        """
        return G/D optimizers
        """

        optim_G = torch.optim.Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=self.hparams["learning_rate"],
            betas=(0.5, 0.999),
        )
        optim_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.hparams["learning_rate"],
            betas=(0.5, 0.999),
        )
        return [optim_G, optim_D], []


def cli_main():

    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=0, type=int)

    # optional... automatically add all the params
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    # setup
    model = Scyclone()
    datamodule = NonParallelSpecDataModule(64)
    logger = pl_loggers.TensorBoardLogger("logs/")
    trainer = pl.Trainer(
        gpus=args.gpus,
        # auto_select_gpus=True,
        max_epochs=400000,  # from Scyclone poster (check my Scyclone summary blog post)
        check_val_every_n_epoch=1500,  # about 1 validation per 10 min
        # reload_dataloaders_every_epoch=True,
        # resume_from_checkpoint="url",
        logger=logger,
    )

    # training
    trainer.fit(model, datamodule=datamodule)

    # testing
    # trainer.test(datamodule=datamodule)


if __name__ == "__main__":  # pragma: no cover
    cli_main()