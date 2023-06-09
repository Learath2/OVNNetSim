from typing import NamedTuple
import numpy as np

class Point2D(NamedTuple):
    x: np.float64
    y: np.float64

def euc_dist(a: Point2D, b: Point2D) -> np.float64:
    return np.linalg.norm(a - b)
