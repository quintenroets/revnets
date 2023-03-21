from ..utils import config
from . import experiments
from .analysis import Analysis


def main():
    runner = Analysis() if config.run_analysis else experiments.Experiment()
    runner = experiments.data_requirements.Runner()
    runner.run()


if __name__ == "__main__":
    main()
