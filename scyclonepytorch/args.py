from argparse import ArgumentParser, Namespace


def parseArgments(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--dir_exp", default=None, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--weights_save_path", default=None, type=str)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--profiler", action="store_true")
    # max: from Scyclone poster (check my Scyclone summary blog post)
    parser.add_argument("--max_epochs", default=400000, type=int)
    # optional... automatically add all the params
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = MNISTDataModule.add_argparse_args(parser)
    parser.add_argument("--noiseless_d", action="store_true")
    return parser.parse_args()
