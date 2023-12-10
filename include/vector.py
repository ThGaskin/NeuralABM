import math
from typing import Sequence, Union

import numpy as np
import torch


# --- The vector class -----------------------------------------------------------------------------------------------
class Vector:
    def __init__(self, x: float, y: float):
        """
        :param x: the x coordinate
        :param y: the y coordinate
        """

        self.x = x
        self.y = y

    # Magic methods ....................................................................................................
    def __abs__(self):
        return math.sqrt(pow(self.x, 2) + pow(self.y, 2))

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y

    def __mod__(self, other):
        return Vector(self.x % other.x, self.y % other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def scalar_mul(self, l: float):
        self.x *= l
        self.y *= l

    def normalise(self, *, norm: float = 1):
        f = norm / self.__abs__()
        self.scalar_mul(f)

    def within_space(self, space: Union[Sequence, "Vector"]) -> bool:
        # Checks whether the vector lies within a square domain
        if isinstance(space, Vector):
            return (0 <= self.x <= space.x) and (0 <= self.y <= space.y)
        else:
            return (space[0][0] <= self.x <= space[0][1]) and (
                space[1][0] <= self.y <= space[1][1]
            )


def distance(
    v: Vector,
    w: Vector,
    *,
    periodic: bool = False,
    space: Union[Vector, Sequence] = None,
    as_tensor: bool = True,
):
    """Returns the distance between two vectors v and w. If the space is periodic, the distance is
    calculated accordingly."""

    if not periodic:
        return (
            abs(v - w) if not as_tensor else torch.tensor(abs(v - w), dtype=torch.float)
        )
    else:
        d = v - w

        if isinstance(space, Vector):
            L_x, L_y = abs(space.x), abs(space.y)
        else:
            L_x, L_y = abs(np.diff(space[0])), abs(np.diff(space[1]))

        dist = math.sqrt(
            pow(min(abs(d.x), L_x - abs(d.x)), 2)
            + pow(min(abs(d.y), L_y - abs(d.y)), 2)
        )

        return dist if not as_tensor else torch.tensor(dist, dtype=torch.float)
