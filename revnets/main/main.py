from .. import experiments


def main():
    experiment_module = experiments.data_requirements
    experiment_module.Experiment().run()


if __name__ == "__main__":
    main()
