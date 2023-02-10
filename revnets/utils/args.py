import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", help="The name of the config file", default=None
    )
    parser.add_argument(
        "--devices", nargs="+", help="Indices of devices to use", default=None
    )
    return parser.parse_args()
