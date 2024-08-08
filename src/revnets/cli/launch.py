from package_utils.context.entry_point import create_entry_point

from revnets.context import context
from revnets.launching import main

entry_point = create_entry_point(main, context)
