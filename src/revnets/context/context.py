from package_utils.context import Context

from revnets.models import Config, Options, Secrets

context = Context(Options, Config, Secrets)
