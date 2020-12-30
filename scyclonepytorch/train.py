import os
from typing import Optional
from argparse import Namespace

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.core.datamodule import LightningDataModule

from .model import Scyclone


def train(args: Namespace, datamodule: LightningDataModule) -> None:
    """Train Scyclone on PyTorch-Lightning.
    """

    ckptAndLogging = CheckpointAndLogging(args.dir_root, args.name_exp, args.name_version)
    # setup
    gpus: int = 1 if torch.cuda.is_available() else 0  # single GPU or CPU
    model = Scyclone(args.sampling_rate, args.noiseless_d)
    ckpt_cb = ModelCheckpoint(period=60, save_last=True, save_top_k=1, monitor="val_loss")
    trainer = pl.Trainer(
        gpus=gpus,
        auto_select_gpus=True,
        precision=32 if args.no_amp else 16,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_interval_epoch,
        # logging/checkpointing
        resume_from_checkpoint=ckptAndLogging.resume_from_checkpoint,
        default_root_dir=ckptAndLogging.default_root_dir,
        checkpoint_callback=ckpt_cb,
        logger=pl_loggers.TensorBoardLogger(
            ckptAndLogging.save_dir, ckptAndLogging.name, ckptAndLogging.version
        ),
        # reload_dataloaders_every_epoch=True,
        profiler=args.profiler,
        progress_bar_refresh_rate=30
    )

    # training
    trainer.fit(model, datamodule=datamodule)


class CheckpointAndLogging:
    """Generate path of checkpoint & logging.

    {dir_root}/
        {name_exp}/
            {name_version}/
                checkpoints/
                    {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
                hparams.yaml
                events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    """

    # [PL's Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api)
    default_root_dir: Optional[str]
    resume_from_checkpoint: Optional[str]
    # [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/logging.html#tensorboard)
    save_dir: str
    name: str
    version: str
    # [PL's ModelCheckpoint callback](https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
    # inferred from above two

    def __init__(
        self,
        dir_root: str,
        name_exp: str = "default",
        name_version: str = "version_-1",
        name_ckpt: str = "last.ckpt",
    ) -> None:

        # ModelCheckpoint
        self.default_root_dir = dir_root
        self.resume_from_checkpoint = os.path.join(dir_root, name_exp, name_version, "checkpoints", name_ckpt)
        # TensorBoardLogger
        self.save_dir = dir_root
        self.name = name_exp
        self.version = name_version


