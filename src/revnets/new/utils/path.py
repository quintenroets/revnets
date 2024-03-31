import plib

root = plib.Path(__file__).parent.parent


class Path(plib.Path):
    assets: plib.Path = plib.Path.assets / root.name
    data = assets / "data"
    config = assets / "config"
    weights = assets / "weights"
    outputs = assets / "outputs"
    results = assets / "results"
