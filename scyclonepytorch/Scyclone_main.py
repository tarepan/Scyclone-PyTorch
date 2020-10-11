import itertools
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from npvcc2016.PyTorch.Lightning.datamodule.waveform import NpVCC2016DataModule
from torch.tensor import Tensor

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

    def __init__(self):
        super().__init__()

        # params
        ## "λcy and λid were set to 10 and 1 in Eq. 1" in Scyclone paper
        ## self.weight_adv = 1 // standard
        self.weight_cycle = 10
        self.weight_identity = 1
        ## "m is a parameter of the hinge loss and is set to (...) 0.5 in our experiments" in Scyclone paper
        self.hinge_offset_D = 0.5
        self.learning_rate = 2.0 * 1e-4
        self.batch_size = 32
        self.save_hyperparameters()

        self.G_A2B = Generator()
        self.G_B2A = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()

        # self.mnist_train = None
        # self.mnist_val = None

    def forward(self, x: Tensor) -> Tensor:
        return self.G_A2B(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Min-Max adversarial training (G:D = 1:1)
        """
        # result = pl.TrainResult(minimize=loss)
        # result.log("train_loss", loss)
        real_A, real_B = batch

        # Generator training
        if optimizer_idx == 0:
            # Generator adversarial losses
            ## hinge loss (from Scyclone paper eq.1)
            """
            change to hinge loss
            """
            fake_B = self.G_A2B(real_A)
            pred_fake_B = self.D_B(fake_B)
            loss_GAN_A2B = torch.mean(F.relu(-1.0 * pred_fake_B))
            fake_A = self.G_B2A(real_B)
            pred_fake_A = self.D_A(fake_A)
            loss_GAN_B2A = torch.mean(F.relu(-1.0 * pred_fake_A))

            # cycle consistency losses
            ## L1 loss (from Scyclone paper eq.1)
            cycled_A = self.G_B2A(fake_B)
            loss_cycle_ABA = F.l1_loss(cycled_A, real_A)
            cycled_B = self.G_A2B(fake_A)
            loss_cycle_BAB = F.l1_loss(cycled_B, real_B)

            # identity mapping losses
            ## L1 loss (from Scyclone paper eq.1)
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

            output = {"loss": loss_G, "log": {"Loss/Generator": loss_G}}
            ## registration for Discriminator loop
            self.fake_B = fake_B
            self.fake_A = fake_A
            ## Is this needed...?
            self.real_B = real_B
            self.real_A = real_A

            # Log to tb
            # if batch_idx % 500 == 0:
            #     self.logger.experiment.add_image(
            #         "Real/A",
            #         make_grid(self.real_A, normalize=True, scale_each=True),
            #         self.current_epoch,
            #     )
            #     self.logger.experiment.add_image(
            #         "Real/B",
            #         make_grid(self.real_B, normalize=True, scale_each=True),
            #         self.current_epoch,
            #     )
            #     self.logger.experiment.add_image(
            #         "Generated/A",
            #         make_grid(self.generated_A, normalize=True, scale_each=True),
            #         self.current_epoch,
            #     )
            #     self.logger.experiment.add_image(
            #         "Generated/B",
            #         make_grid(self.generated_B, normalize=True, scale_each=True),
            #         self.current_epoch,
            #     )
            return output

        # Discriminator training
        if optimizer_idx == 1:
            m = self.hinge_offset_D

            ## ones:  torch.ones( pred_real.shape).type_as(pred_real)
            ## zeros: torch.zeros(pred_fake.shape).type_as(pred_fake)

            # Adversarial loss: hinge loss (from Scyclone paper eq.1)
            # D_A
            ## Real loss
            ## [todo] edge 16frame cut
            pred_A_real = self.D_A(real_A)
            loss_D_A_real = torch.mean(F.relu(m - pred_A_real))
            ## Fake loss
            ## [todo] edge 16frame cut
            pred_A_fake = self.D_A(self.fake_A.detach())
            loss_D_A_fake = torch.mean(F.relu(m + pred_A_fake))
            ## D_A total loss
            loss_D_A = loss_D_A_real + loss_D_A_fake

            # D_B
            ## Real loss
            ## [todo] edge 16frame cut
            pred_B_real = self.D_B(real_B)
            loss_D_B_real = torch.mean(F.relu(m - pred_B_real))
            ## Fake loss
            ## [todo] edge 16frame cut
            pred_B_fake = self.D_B(self.fake_B.detach())
            loss_D_B_fake = torch.mean(F.relu(m + pred_B_fake))
            ## D_B total loss
            loss_D_B = loss_D_B_real + loss_D_B_fake

            # Total
            loss_D = loss_D_A + loss_D_B
            output = {"loss": loss_D, "log": {"Loss/Discriminator": loss_D}}
            return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        result.log("val_acc", accuracy(y_hat, y))
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("test_loss", loss)
        result.log("test_acc", accuracy(y_hat, y))
        return result

    def configure_optimizers(self):
        """
        return G/D optimizers
        """

        optim_G = torch.optim.Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999),
        )
        optim_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.hparams.learning_rate,
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
    model = Scyclone(**vars(args))
    datamodule = NpVCC2016DataModule(64, download=True)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=2, limit_train_batches=200)

    # training
    trainer.fit(model, datamodule=datamodule)

    # testing
    trainer.test(datamodule=datamodule)


if __name__ == "__main__":  # pragma: no cover
    cli_main()