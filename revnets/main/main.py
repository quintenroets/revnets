from ..utils import get_args, launch_experiments
from .experiment import Experiment


def main():
    args = get_args()
    if args.config_name:
        Experiment().run()
    else:
        launch_experiments(args.devices)


if __name__ == "__main__":
    main()
