from .. import experiments
from ..utils import get_args


def main():
    args = get_args()
    experiment_module = (
        experiments.__dict__[args.experiment]
        if args.experiment
        else experiments.experiment
    )
    experiment_module.Experiment().run()


if __name__ == "__main__":
    main()
