import sys
from os.path import dirname as up

import numpy as np
import pytest
import xarray as xr
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

ops = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="model_plots.data_ops"
)

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/data_ops.yml")
test_cfg = load_yml(CFG_FILENAME)

# Generate an example dataset consisting of various x-values and associated loss values
ds_0 = xr.Dataset(
    data_vars=dict(alpha=(["x"], np.random.rand(1)), p=(["x"], np.random.rand(1))),
    coords=dict(x=("x", np.arange(1))),
)
ds_0["p"] /= np.sum(ds_0["p"])

ds_1 = xr.Dataset(
    data_vars=dict(alpha=(["x"], np.random.rand(10)), p=(["x"], np.random.rand(10))),
    coords=dict(x=("x", np.arange(10))),
)
ds_1["p"] /= np.sum(ds_1["p"])

x = y = z = np.linspace(-2, 2, 10)
ds_2 = xr.Dataset(
    data_vars=dict(
        alpha=(
            ["x", "y", "z"],
            np.squeeze([[[[np.sin(i * j * k)] for i in x] for j in y] for k in z]),
        ),
        p=(["x", "y", "z"], np.random.rand(len(x), len(y), len(z))),
    ),
    coords=dict(x=("x", x), y=("y", y), z=("z", z)),
)
ds_2["p"] /= np.sum(ds_2["p"])

ds_s = xr.Dataset(
    data_vars=dict(
        alpha=(
            ["x", "y", "z"],
            np.squeeze([[[[np.sin(i * j * k)] for i in x] for j in y] for k in z]),
        ),
        p=(["x", "y", "z"], np.random.rand(len(x), len(y), len(z))),
    ),
    coords=dict(
        x=("x", ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        y=("y", y),
        z=("z", z),
    ),
)
ds_s["p"] /= np.sum(ds_s["p"])


def test_apply_along_dim():

    m1 = ops.mean(ds_1, x="alpha", p="p")
    assert m1
    assert len(m1.dims) == 0

    m1 = ops.mean(ds_2, x="alpha", p="p", along_dim=["x"])
    assert len(m1.dims) == 2
    assert m1.equals(ops.mean(ds_2, x="alpha", p="p", exclude_dim=["y", "z"]))

    m1 = ops.mean(ds_s, x="alpha", p="p", along_dim=["y"])
    assert m1


# ----------------------------------------------------------------------------------------------------------------------
# BASIC STATISTICS FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def test_mean_std():
    for func in [ops.mean, ops.std]:
        # Calculate the indicator for a point
        m1 = func(ds_0, x="alpha", p="p")
        assert m1

        # Calculate the indicator for a one-dimensional dataset
        m1 = func(ds_1, x="alpha", p="p")
        assert m1
        m1 = func(ds_1, x="alpha", p="p", along_dim=["x"])
        assert m1

        # Calculate the indicator along one dimension
        m1 = func(ds_2, x="alpha", p="p", along_dim=["x"])
        assert m1

        # Calculate the indicator along two dimensions
        m1 = func(ds_2, x="alpha", p="p", along_dim=["x", "y"])
        assert m1


def test_mode():
    # Calculate the mode for the trivial dataset
    m1 = ops.mode(ds_0, p="p")
    assert m1

    # Calculate the mode for the one-dimensional dataset
    m1 = ops.mode(ds_1, p="p")
    assert m1
    m1 = ops.mode(ds_1, p="p", along_dim=["x"])
    assert m1

    # Calculate the mode along a particular dimension
    m1 = ops.mode(ds_2, along_dim=["x"], p="p")
    assert m1


def test_avg_peak_width():
    # Calculate the peak widths along all dimensions
    m1 = ops.avg_peak_width(
        ds_2["alpha"], along_dim=["x", "y", "z"], width=[None, None]
    )
    assert m1
    assert len(m1.dims) == 0
    assert m1.equals(ops.avg_peak_width(ds_2["alpha"], width=[None, None]))

    # Calculate the peak widths along two dimensions
    m1 = ops.avg_peak_width(ds_2["alpha"], along_dim=["x", "y"], width=[None, None])
    assert m1
    assert len(m1.dims) == 1

    # Calculate the peak widths along one dimension
    m1 = ops.avg_peak_width(ds_2["alpha"], along_dim=["x"], width=[None, None])
    assert m1
    assert len(m1.dims) == 2


def test_p_value():
    # Calculate the p-value along one dimension
    m1 = ops.p_value(ds_2, along_dim=["x"], x="alpha", p="p", t=0.0)
    assert m1
    assert [0 <= p <= 1 for p in m1["p_value"].data.flatten()]


# ----------------------------------------------------------------------------------------------------------------------
# MARGINAL DENSITY TEST
# ----------------------------------------------------------------------------------------------------------------------
def test_marginal_density():
    # Calculate the marginals along all dimensions
    m1 = ops.marginals(ds_2, x="alpha", p="p")
    assert m1
    assert len(m1.dims) == 1

    # Check the probabilities are normalised
    assert float(
        m1["p"].sum() * (abs(m1["x"][1] - m1["x"][0])).item()
    ) == pytest.approx(1, 1e-5)

    # Calculate the marginals along one dimension
    m1 = ops.marginals(ds_2, x="alpha", p="p", along_dim=["x"])
    assert m1
    assert len(m1.dims) == 3

    # Calculate the marginals along two dimensions
    m1 = ops.marginals(ds_2, x="alpha", p="p", along_dim=["x", "y"])
    assert m1
    assert len(m1.dims) == 2
