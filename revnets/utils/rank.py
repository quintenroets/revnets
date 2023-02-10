from collections.abc import Callable
from functools import wraps
from typing import Any

from pytorch_lightning.utilities import rank_zero

RANK = rank_zero.rank_zero_only.rank  # need to fix rank here


def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any | None:
        result = fn(*args, **kwargs) if RANK == 0 else None
        return result

    return wrapped_fn
