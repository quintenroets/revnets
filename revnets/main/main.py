from .. import experiments
from ..utils import get_args


def main():
    args = get_args()
    experiment_name = args.experiment or "data_requirements"
    experiment_module = experiments.__dict__[experiment_name]
    experiment_module.Experiment().run()


if __name__ == "__main__":
    main()
