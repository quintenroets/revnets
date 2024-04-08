from package_utils.context import Context

from ...models import Config, Options

context = Context[Options, Config, None](Options, Config, None)
