from package_utils.context.entry_point import create_entry_point

from revnets.launching import main

from ..context import context

entry_point = create_entry_point(main, context)
