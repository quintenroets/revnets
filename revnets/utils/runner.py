import cli


def run():
    seeds = range(3)
    # seeds = range(3, 6)
    # seeds = range(7, 10)
    seeds = range(10)

    for seed in seeds:
        args = {"config-name": "mininet_images_small", "seed": seed}
        cli.run("revnets", args)
