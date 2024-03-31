from .. import experiments
from ..utils import config, get_args


def main() -> None:
    args = get_args()
    if args.seed_range:
        for seed in range(args.seed_range):
            config.manual_seed = seed
            run(args)
    else:
        run(args)


def run(args) -> None:
    experiment_module = get_experiment_module(args)
    experiment_module.Experiment().run()


def get_experiment_module(args):
    if args.experiment:
        experiment_module = experiments
        analysis_keyword = "_analysis"
        if analysis_keyword in args.experiment:
            args.experiment = args.experiment.replace(analysis_keyword, "")
            experiment_module = experiments.analysis
        experiment_module = getattr(experiment_module, args.experiment)
    else:
        experiment_module = experiments.experiment
    return experiment_module


if __name__ == "__main__":
    main()
