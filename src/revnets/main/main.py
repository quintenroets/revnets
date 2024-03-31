from ..context import context


def main() -> None:
    """
    Python package template.
    """
    message = "main functionality"
    if context.options.debug:
        print(message)
    print(context.secrets)
