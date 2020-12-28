from argparse import ArgumentParser

import pytorch_lightning as pl

from .train import train
from .datamodule import DataLoaderPerformance, NonParallelSpecDataModule
from .args import parseArgments


def main_train():

    # Random seed
    pl.seed_everything(1234)

    # Args
    parser = ArgumentParser()
    args_scpt = parseArgments(parser)

    # Datamodule
    loader_perf = DataLoaderPerformance(args_scpt.num_workers, not args_scpt.no_pin_memory)
    datamodule = NonParallelSpecDataModule(64, loader_perf)

    # Train
    train(args_scpt, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()