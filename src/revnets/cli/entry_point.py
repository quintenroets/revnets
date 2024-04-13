from package_utils.context.entry_point import create_entry_point

from revnets.main.main import Experiment

from ..context import context

experiment = Experiment()

entry_point = create_entry_point(experiment.run, context)
