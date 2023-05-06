import numpy as np
from matplotlib import cm


def get_colors(n=10):
    mappers = {
        10: (cm.tab10,),
        20: (cm.tab20,),
        40: (cm.tab20, cm.tab20b),
        60: (cm.tab20, cm.tab20b, cm.tab20c),
    }
    color_maps = None
    for number in mappers:
        if n <= number:
            n = number
            color_maps = mappers[number]
            break
    if color_maps is None:
        color_maps = list(mappers[60]) * (n // 60 + 1)

    n_color = 10 if n <= 10 else 20
    colors = [
        color for color_map in color_maps for color in get_points(color_map, n_color)
    ]
    return colors


def get_points(color_map, n):
    color_points = np.linspace(0, 1, n)
    return color_map(color_points)
