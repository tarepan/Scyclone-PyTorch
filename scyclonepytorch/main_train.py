from argparse import ArgumentParser

import pytorch_lightning

from .args import parseArgments
from .datamodule import DataLoaderPerformance, NonParallelSpecDataModule
from .train import train


def main_train():
    """Train Scyclone-PyTorch with cli arguments and the default dataset.
    """

    # Random seed
    pytorch_lightning.seed_everything(1234)

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