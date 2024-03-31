import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        dest="config_name",
        help="The name of the config file",
        default=None,
    )
    parser.add_argument(
        "--experiment", help="The name of the experiment to run", default=None
    )
    parser.add_argument(
        "--seed", help="The manual seed of the experiment to run", default=None
    )
    parser.add_argument(
        "--seed-range",
        dest="seed_range",
        help="The manual seed of the experiment to run",
        default=None,
    )
    parser.add_argument("--devices", nargs="+", help="Indices of devices to use")
    parser_args = ([],) if "pytest" in sys.modules else ()
    args, _ = parser.parse_known_args(*parser_args)
    return args
