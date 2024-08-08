from package_utils.context.entry_point import create_entry_point

from revnets.context import context
from revnets.main.main import Experiment

experiment = Experiment()

entry_point = create_entry_point(experiment.run, context)
