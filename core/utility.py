from typing import NamedTuple, TypeVar
import numpy as np
import numpy.typing as npt

class Point2D(NamedTuple):
    x: np.float64
    y: np.float64

def euc_dist(a: Point2D, b: Point2D) -> np.float64:
    return np.linalg.norm(np.array(a) - np.array(b))

# These use 10log10 as we are dealing with power
def lin2db(a: npt.ArrayLike) -> npt.ArrayLike:
    return 10 * np.log10(a)

def db2lin(a: npt.ArrayLike) -> npt.ArrayLike:
    return np.power(10, np.asarray(a) / 10)
