import math
import sys
from os.path import dirname as up

import numpy as np
import pytest
import torch
from dantro._import_tools import import_module_from_path

sys.path.insert(0, up(up(up(__file__))))

vec = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include.vector")
Vector = vec.Vector
distance = vec.distance

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

vectors = [Vector(2, 3), Vector(-1, -4), Vector(0.4, -0.2), Vector(0.0, 0)]
non_zero_vectors = [Vector(2, 3), Vector(-1, -4), Vector(0.4, -0.2)]
zero_vector = Vector(0, 0)
unit_vector = Vector(1, 1)


def test_vector():
    """Tests the Vector class"""

    for v in vectors:
        assert v == v
        assert v != Vector(-1, -1)
        assert v + zero_vector == v

        assert math.sqrt(v * v) == abs(v)
        assert v * zero_vector == 0

        assert -(-v) == v
        assert (abs(-v)) == abs(v)

    assert abs(unit_vector) == math.sqrt(2)
    assert abs(zero_vector) == 0

    assert vectors[0] + vectors[1] == Vector(1, -1)
    assert vectors[0] - vectors[1] == Vector(3, 7)

    q = Vector(0.0, 0.0)
    q.scalar_mul(2.3)
    assert q == zero_vector

    q = Vector(1.0, 1.0)
    q.scalar_mul(-math.pi)
    assert q == Vector(-math.pi, -math.pi)

    for v in non_zero_vectors:
        v.normalise()
        assert abs(v) == pytest.approx(1, 1e-8)

        v.normalise(norm=2.5)
        assert abs(v) == pytest.approx(2.5, 1e-8)

    space_list = [[-10, 10], [-10, 10]]
    space_np = np.array(space_list)
    space_torch = torch.from_numpy(space_np)
    space_slice = [[-20, -10], [-10, -10]]
    space_small = [[-20, -10], [-10, -9]]
    for v in vectors:
        assert v.within_space(space_list)
        assert v.within_space(space_np)
        assert v.within_space(space_torch)
        assert not v.within_space(space_slice)
        assert not v.within_space(space_small)

    space = Vector(3, 5)
    assert zero_vector.within_space(space)


def test_distances():
    """Tests the distance function"""

    # Non-periodic case
    for v in vectors:
        assert distance(v, zero_vector) == abs(v)
        assert distance(zero_vector, v) == abs(v)
        d_tensor = distance(v, zero_vector, as_tensor=True)
        assert d_tensor == abs(v)
        assert torch.is_tensor(d_tensor)

    # Periodic case
    large_space = [[-100, 100], [-100, 100]]
    for v in vectors:
        assert distance(v, zero_vector, periodic=True, space=large_space) == abs(v)
        assert distance(zero_vector, v, periodic=True, space=large_space) == abs(v)

    # Boundary points all 0 distance from each other
    small_space = [[-2, 2], [-2, 2]]
    q, p, r, t = Vector(2, 2), Vector(-2, -2), Vector(-2, 2), Vector(2, -2)
    for v in [p, q, r, t]:
        for w in [p, q, r, t]:
            assert distance(v, w, periodic=True, space=small_space) == 0
            assert distance(
                v, zero_vector, periodic=True, space=small_space
            ) == distance(v, zero_vector)

    # Test specific value on torus
    v, w = Vector(-1.5, -1.5), Vector(1.5, 1.5)
    assert distance(v, w, periodic=True, space=small_space) == math.sqrt(2)
