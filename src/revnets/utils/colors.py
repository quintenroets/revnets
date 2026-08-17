from collections.abc import Iterator

import numpy as np
from matplotlib import cm
from matplotlib.colors import Colormap
from numpy.typing import NDArray

SMALL_NUMBER_OF_COLORS = 10


def get_colors(
    number_of_colors: int = SMALL_NUMBER_OF_COLORS,
) -> list[NDArray[np.float64]]:
    return list(generate_colors(number_of_colors))


def generate_colors(number_of_colors: int) -> Iterator[NDArray[np.float64]]:
    number_of_points, color_maps = next(generate_color_maps(number_of_colors))
    number_of_points = (
        SMALL_NUMBER_OF_COLORS if number_of_points <= SMALL_NUMBER_OF_COLORS else 20
    )
    for color_map in color_maps:
        points = np.linspace(0, 1, number_of_points)
        color_points = color_map(points)
        yield from color_points


def generate_color_maps(
    number_of_colors: int,
) -> Iterator[tuple[int, tuple[Colormap, ...]]]:
    mappers = {
        10: (cm.tab10,),
        20: (cm.tab20,),
        40: (cm.tab20, cm.tab20b),
        60: (cm.tab20, cm.tab20b, cm.tab20c),
    }
    for number, color_map in mappers.items():
        if number_of_colors <= number:
            yield number_of_colors, color_map
    color_maps = list(mappers[60]) * (number_of_colors // 60 + 1)  # pragma: nocover
    yield number_of_colors, tuple(color_maps)  # pragma: nocover
