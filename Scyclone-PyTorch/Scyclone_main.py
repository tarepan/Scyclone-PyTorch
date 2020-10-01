import itertools
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from modules import Generator, Discriminator

"""
G: Generator
D: Discriminator
A2B: map A to B (c.f. B2A)
"""
# inspired by CycleGAN family https://github.com/HasnainRaz/Fast-AgingGAN/blob/master/gan_module.py


class Scyclone(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=128,
        learning_rate=2.0 * 1e-4,
        batch_size=32,
        num_workers=4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.G_A2B = Generator(hparams["ngf"], n_residual_blocks=hparams["n_blocks"])
        self.G_B2A = Generator(hparams["ngf"], n_residual_blocks=hparams["n_blocks"])
        self.D_A = Discriminator(hparams["ndf"])
        self.D_B = Discriminator(hparams["ndf"])

        self.mnist_train = None
        self.mnist_val = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # result = pl.TrainResult(minimize=loss)
        # result.log("train_loss", loss)
        real_A, real_B = batch

        # Generator training
        if optimizer_idx == 0:
            # identity mapping losses
            ## L1 loss (from Scyclone paper eq.1)
            weight_identity = self.hparams["identity_weight"]
            same_B = self.G_A2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * weight_identity
            same_A = self.G_B2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * weight_identity

            # Generator adversarial loss
            """
            change to hinge loss
            """
            fake_B = self.G_A2B(real_A)
            pred_fake = self.D_B(fake_B)
            loss_GAN_A2B = (
                F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake))
                * self.hparams["adv_weight"]
            )
            fake_A = self.G_B2A(real_B)
            pred_fake = self.D_A(fake_A)
            loss_GAN_B2A = (
                F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake))
                * self.hparams["adv_weight"]
            )

            # cycle consistency loss
            ## L1 loss (from Scyclone paper eq.1)
            weight_cycle = self.hparams["cycle_weight"]
            cycled_A = self.G_B2A(fake_B)
            loss_cycle_ABA = F.l1_loss(cycled_A, real_A) * weight_cycle
            cycled_B = self.G_A2B(fake_A)
            loss_cycle_BAB = F.l1_loss(cycled_B, real_B) * weight_cycle

            # Total loss
            g_loss = (
                loss_identity_A
                + loss_identity_B
                + loss_GAN_A2B
                + loss_GAN_B2A
                + loss_cycle_ABA
                + loss_cycle_BAB
            )

            output = {"loss": g_loss, "log": {"Loss/Generator": g_loss}}
            self.generated_B = fake_B
            self.generated_A = fake_A

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
            # D_A
            ## Real loss
            pred_real = self.D_A(real_A)
            loss_D_real = F.mse_loss(
                pred_real, torch.ones(pred_real.shape).type_as(pred_real)
            )
            ## Fake loss
            fake_A = self.generated_A
            pred_fake = self.D_A(fake_A.detach())
            loss_D_fake = F.mse_loss(
                pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake)
            )
            ## D_A total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # D_B
            ## Real loss
            pred_real = self.D_B(real_B)
            loss_D_real = F.mse_loss(
                pred_real, torch.ones(pred_real.shape).type_as(pred_real)
            )
            ## Fake loss
            fake_B = self.generated_B
            pred_fake = self.D_B(fake_B.detach())
            loss_D_fake = F.mse_loss(
                pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake)
            )
            ## D_B total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            # Total
            d_loss = loss_D_A + loss_D_B
            output = {"loss": d_loss, "log": {"Loss/Discriminator": d_loss}}
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))
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
    from project.datasets.mnist import mnist

    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=0, type=int)

    # optional... automatically add all the params
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # data
    mnist_train, mnist_val, test_dataset = mnist()

    # model
    model = Scyclone(**vars(args))

    # training
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=2, limit_train_batches=200)
    trainer.fit(model, mnist_train, mnist_val)

    trainer.test(test_dataloaders=test_dataset)


if __name__ == "__main__":  # pragma: no cover
    cli_main()