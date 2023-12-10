import sys
from os.path import dirname as up

import numpy as np
import pytest
import scipy.integrate
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


# ----------------------------------------------------------------------------------------------------------------------
# DECORATOR
# ----------------------------------------------------------------------------------------------------------------------


def test_apply_along_dim():
    """Tests the apply_along_dim decorator"""

    @ops.apply_along_dim
    def _test_func(data):
        return xr.DataArray(np.mean(data), name=data.name)

    # Test on DataArray
    da = xr.DataArray(
        np.ones((10, 10, 10)),
        dims=["x", "y", "z"],
        coords={k: np.arange(10) for k in ["x", "y", "z"]},
        name="foo",
    )

    res = _test_func(da, along_dim=["x"])

    # apply_along_dim returns a Dataset
    assert isinstance(res, xr.Dataset)

    # Assert operation was applied along that dimension
    assert set(res.coords.keys()) == {"y", "z"}
    assert list(res.data_vars) == ["foo"]
    assert all(res["foo"].data.flatten() == 1)

    # Assert specifying operation is the same as excluding certain dimensions
    assert res == _test_func(da, exclude_dim=["y", "z"])

    # Apply along all dimensions
    res = _test_func(da, along_dim=["x", "y", "z"])
    assert res == xr.Dataset(data_vars=dict(foo=([], 1)))

    # Test applying function without decorator
    res = _test_func(da)
    assert isinstance(res, type(da))
    assert res == xr.DataArray(1)

    # Test applying the decorator along multiple args
    @ops.apply_along_dim
    def _test_func_nD(da1, da2, da3, *, op: str = "sum"):
        if op == "sum":
            return (da1 * da2 * da3).sum()
        else:
            return (da1 * da2 * da3).mean()

    res = _test_func_nD(da, 2 * da, -1 * da, along_dim=["x"])
    assert set(res.coords.keys()) == {"y", "z"}
    assert all(res["foo"].data.flatten() == len(da.coords["x"]) * 2 * -1)
    assert res == _test_func_nD(da, 2 * da, -1 * da, exclude_dim=["y", "z"])

    # Test passing kwargs to function
    res = _test_func_nD(da, 2 * da, -1 * da, along_dim=["x"], op="mean")
    assert all(res["foo"].data.flatten() == 1 * 2 * -1)

    # Test passing both ``along_dim`` and ``exclude_dim`` raises an error
    with pytest.raises(ValueError, match="Cannot provide both"):
        _test_func(da, along_dim=["x"], exclude_dim=["y"])


# ----------------------------------------------------------------------------------------------------------------------
# DATA RESHAPING AND REORGANIZING
# ----------------------------------------------------------------------------------------------------------------------
def test_concat():
    """Tests concatenation of multiple xarray objects"""

    da = xr.DataArray(
        np.ones((10, 10, 10)),
        dims=["x", "y", "z"],
        coords={k: np.arange(10) for k in ["x", "y", "z"]},
        name="foo",
    )

    # Test on DataArrays and Datasets
    for _obj in [da, xr.Dataset(dict(foo=da))]:
        res = ops.concat([_obj, _obj, _obj], "type", ["a", "b", "c"])
        assert set(res.coords.keys()) == set(list(_obj.coords.keys()) + ["type"])


def test_flatten_dims():
    """Test flattening coordinates of xarray objects into one"""

    da = xr.DataArray(
        np.ones((5, 5, 5)),
        dims=["dim1", "dim2", "dim3"],
        coords={k: np.arange(5) for k in ["dim1", "dim2", "dim3"]},
    )

    # Test on DataArrays and Datasets
    for _obj in [da, xr.Dataset(dict(foo=da))]:
        res = ops.flatten_dims(_obj, dims={"dim2": ["dim2", "dim3"]})
        assert set(res.coords.keys()) == {"dim1", "dim2"}
        assert len(
            res.coords["dim2"] == len(_obj.coords["dim2"]) * len(_obj.coords["dim3"])
        )

        # Test reassigning coordinates
        res = ops.flatten_dims(
            _obj, dims={"dim2": ["dim2", "dim3"]}, new_coords=np.arange(25, 50, 1)
        )
        assert all(res.coords["dim2"] == np.arange(25, 50, 1))


def test_broadcast():
    """Test broadcasting xr.DataArray s into a single Dataset"""
    da1 = xr.DataArray(np.random.rand(10, 3), dims=["sample", "parameter"])
    da2 = xr.DataArray(np.exp(-np.linspace(0, 1, 10)), dims=["sample"])
    res = ops.broadcast(da1, da2, x="x", p="loss")

    assert isinstance(res, xr.Dataset)
    assert set(res.data_vars) == {"x", "loss"}
    assert res == ops.broadcast(da2, da1)


# ----------------------------------------------------------------------------------------------------------------------
# BASIC STATISTICS FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def test_stat_function():
    """Tests the statistics functions on a normal distribution"""
    _x = np.linspace(-5, 5, 1000)
    _m, _std = 0.0, 1.0
    _f = np.exp(-((_x - _m) ** 2) / (2 * _std**2))
    _norm = scipy.integrate.trapezoid(_f, _x)

    ds = xr.Dataset(dict(y=(["x"], _f)), coords=dict(x=_x))

    # Test normalisation
    ds = ops.normalize(ds, x="x", y="y")
    assert scipy.integrate.trapezoid(ds["y"], ds.coords["x"]) == pytest.approx(
        1.0, 1e-4
    )

    # Test mean
    mean = ops.stat_function(ds, stat="mean", x="x", y="y")
    assert mean == pytest.approx(_m, abs=1e-3)

    # Test mean with x as a coordinate or variable
    ds = xr.Dataset(dict(y=(["x"], ds["y"].data), x_val=ds.coords["x"]))
    assert mean == ops.stat_function(ds, stat="mean", x="x_val", y="y")
    assert mean == ops.stat_function(ds, stat="mean", x="x", y="y")

    # Test standard deviation
    std = ops.stat_function(ds, stat="std", x="x", y="y")
    assert std == pytest.approx(_std, abs=1e-3)

    # Test interquartile range
    iqr = ops.stat_function(ds, stat="iqr", x="x", y="y")
    assert iqr == pytest.approx(1.34, abs=1e-2)

    # Test mode
    mode = ops.stat_function(ds, stat="mode", x="x", y="y")
    assert mode["mode_x"].data.item() == pytest.approx(_m, abs=1e-2)
    assert mode["mode_y"].data.item() == pytest.approx(1.0 / _norm, abs=1e-2)

    # Test peak width calculation
    peak_widths = ops.stat_function(ds, stat="avg_peak_width", x="x", y="y", width=1)
    assert peak_widths["mean_peak_width"] == pytest.approx(2.355 * _std, abs=1e-2)
    assert peak_widths["peak_width_std"] == 0.0

    # Test p-value calculation
    assert ops.p_value(ds, 0.0, x="x", y="y") == pytest.approx(0.5, abs=1e-1)
    assert ops.p_value(ds, -1, x="x", y="y") == pytest.approx(0.159, abs=5e-3)
    assert ops.p_value(ds, -1, x="x", y="y") == pytest.approx(
        ops.p_value(ds, +1, x="x", y="y"), abs=5e-3
    )
    assert ops.p_value(ds, xr.DataArray(0.0), x="x", y="y") == ops.p_value(
        ds, 0.0, x="x", y="y"
    )

    # Assert the p-value for a Gaussian wrt the mean is the same as wrt to the mode
    assert ops.p_value(ds, -1, x="x", y="y", null="mean") == ops.p_value(
        ds, -1, x="x", y="y", null="mode"
    )


# ----------------------------------------------------------------------------------------------------------------------
# HISTOGRAMS
# ----------------------------------------------------------------------------------------------------------------------
def test_hist():
    """Tests histogram functions"""

    _n_samples = 100
    _n_vals = 100
    _n_bins = 20

    # Test histogramming a 1D array
    da = xr.DataArray(
        np.random.rand(_n_vals), dims=["i"], coords=dict(i=np.arange(_n_vals))
    )
    hist = ops.hist(da, _n_bins, dim="i")
    assert set(hist.coords.keys()) == {"bin_idx"}
    assert set(hist.data_vars) == {"count", "x"}

    # Test total number of counts has not changed
    assert all(hist["count"].sum("bin_idx").data.flatten() == _n_vals)

    # Repeat the same thing, this time using the bin centres as coordinates
    hist = ops.hist(da, _n_bins, [0, 1], dim="i", use_bins_as_coords=True)
    assert isinstance(hist, xr.DataArray)
    assert set(hist.coords.keys()) == {"x"}
    assert all(
        hist.coords["x"].data
        == 0.5
        * (np.linspace(0, 1, _n_bins + 1)[1:] + np.linspace(0, 1, _n_bins + 1)[:-1])
    )
    assert all(hist.sum("x").data.flatten() == _n_vals)

    # Test histogramming a 2D array
    da = xr.DataArray(
        np.random.rand(_n_samples, _n_vals),
        dims=["sample", "i"],
        coords=dict(sample=np.arange(_n_samples), i=np.arange(_n_vals)),
    )
    hist = ops.hist(da, _n_bins, dim="i")
    assert set(hist.coords.keys()) == {"sample", "bin_idx"}
    assert set(hist.data_vars) == {"count", "x"}
    assert all(hist["count"].sum("bin_idx").data.flatten() == _n_vals)

    hist = ops.hist(da, _n_bins, dim="i", ranges=[0, 1], use_bins_as_coords=True)
    assert set(hist.coords.keys()) == {"sample", "x"}
    assert all(hist.sum("x").data.flatten() == _n_vals)

    # Test histogramming a 3D array
    da = da.expand_dims(dict(dim0=np.arange(4)))
    hist = ops.hist(
        da,
        dim="i",
        exclude_dim=["dim0"],
        bins=_n_bins,
        ranges=[0, 1],
        use_bins_as_coords=True,
    )
    assert set(hist.coords.keys()) == {"sample", "x", "dim0"}

    hist = ops.hist(da, dim="i", exclude_dim=["dim0"], bins=_n_bins)
    assert set(hist.coords.keys()) == {"sample", "bin_idx", "dim0"}

    # Test normalisation of bin counts
    hist_normalised = ops.hist(da, bins=100, dim="i", ranges=[0, 1], normalize=2.0)
    assert hist_normalised["count"].sum("bin_idx").data.flatten() == pytest.approx(
        2.0, abs=1e-10
    )

    # Test selecting range
    hist_clipped = ops.hist(da, bins=100, dim="i", ranges=[1, 2])
    assert all(hist_clipped["count"].data.flatten() == 0)


# ----------------------------------------------------------------------------------------------------------------------
# PROBABILITY DENSITY OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


def test_joint_and_marginal_2D():
    """Test two-dimensional joint distributions and marginals from joints are correctly calculated"""

    # Generate a 2D-Gaussian on a square domain
    _lower, _upper, _n_vals = -5, +5, 1000
    _bins = 50
    _m, _std = 0.0, 1.0
    _x, _y = (_upper - _lower) * np.random.rand(_n_vals) + _lower, (
        _upper - _lower
    ) * np.random.rand(_n_vals) + _lower
    _f = np.exp(-((_x - _m) ** 2) / (2 * _std**2)) * np.exp(
        -((_y - _m) ** 2) / (2 * _std**2)
    )

    # Calculate the joint distribution
    joint = ops.joint_2D(_x, _y, _f, bins=_bins, normalize=1.0)
    assert set(joint.coords.keys()) == {"x", "y"}

    # Assert the maximum of the joint is roughly in the middle
    assert all(
        [
            idx == pytest.approx(25, abs=2)
            for idx in np.unravel_index(joint.argmax(), joint.shape)
        ]
    )

    # Assert the marginal distribution of each dimension is normalized
    marginal_x = ops.marginal_from_joint(joint, parameter="x")
    marginal_y = ops.marginal_from_joint(joint, parameter="y")
    assert scipy.integrate.trapezoid(marginal_x["y"], marginal_x["x"]) == pytest.approx(
        1.0, abs=1e-4
    )
    assert scipy.integrate.trapezoid(marginal_y["y"], marginal_y["x"]) == pytest.approx(
        1.0, abs=1e-4
    )

    # Assert the marginals are again approximately Gaussian
    assert ops.mean(marginal_x, x="x", y="y") == pytest.approx(0.0, abs=1e-1)
    assert ops.mean(marginal_y, x="x", y="y") == pytest.approx(0.0, abs=1e-1)
    assert ops.std(marginal_x, x="x", y="y") == pytest.approx(1.0, abs=5e-2)
    assert ops.std(marginal_y, x="x", y="y") == pytest.approx(1.0, abs=5e-2)

    # Assert alternative joint operation does the same thing
    joint_from_ds = ops.joint_2D_ds(
        xr.Dataset(dict(x=_x, y=_y)), _f, x="x", y="y", bins=_bins, normalize=1.0
    )
    assert joint_from_ds.equals(joint)

    # Assert 3D joint
    _z = (_upper - _lower) * np.random.rand(_n_vals) + _lower
    samples = (
        ops.concat(
            [xr.DataArray(_x), xr.DataArray(_y), xr.DataArray(_z)],
            "parameter",
            ["a", "b", "c"],
        )
        .transpose()
        .assign_coords(dict(dim_0=np.arange(_n_vals)))
    )

    _f = xr.DataArray(_f * np.exp(-((_z - _m) ** 2) / (2 * _std**2))).assign_coords(
        dict(dim_0=np.arange(_n_vals))
    )

    joint_3D = ops.joint_DD(samples, _f, bins=50)
    assert set(joint_3D.coords) == {"a", "b", "c"}


def test_distances_between_densities():
    """Tests the Hellinger distance between distributions"""
    _x = np.linspace(-5, 5, 500)
    Gaussian = xr.DataArray(
        np.exp(-(_x**2) / 2), dims=["x"], coords=dict(x=_x)
    )  # mean = 0, std = 1
    Gaussian /= scipy.integrate.trapezoid(Gaussian.data, _x)

    assert ops.Hellinger_distance(Gaussian, Gaussian) == 0
    assert ops.relative_entropy(Gaussian, Gaussian) == 0
    assert ops.Lp_distance(Gaussian, Gaussian, p=2) == 0

    # Test calculating the distances on a xr.Dataset instead
    Gaussian = xr.Dataset(dict(_x=Gaussian.coords["x"], y=Gaussian))
    assert ops.Hellinger_distance(Gaussian, Gaussian, x="_x", y="y") == 0
    Gaussian = xr.Dataset(dict(y=Gaussian["y"]))
    assert ops.Hellinger_distance(Gaussian, Gaussian, x="x", y="y") == 0

    _x = np.linspace(0, 5, 500)
    Uniform1 = xr.DataArray(
        np.array([1 if 1 <= x <= 3 else 0 for x in _x]), dims=["x"], coords=dict(x=_x)
    )
    Uniform2 = xr.DataArray(
        np.array([1 if 2 <= x <= 4 else 0 for x in _x]), dims=["x"], coords=dict(x=_x)
    )

    # Total area where the two do not overlap is 2, so Hellinger distance = 1/2 * 2 = 1
    assert ops.Hellinger_distance(Uniform1, Uniform2) == pytest.approx(1.0, abs=1e-2)

    # Test interpolation works: shifted uniform distribution, area of non-overlap = 3, so Hellinger
    # distance = 3/2
    Uniform1 = xr.DataArray(
        np.array([1 if 2 <= x <= 4 else 0 for x in np.linspace(1, 6, 500)]),
        dims=["x"],
        coords=dict(x=np.linspace(1, 6, 500)),
    )
    Uniform2 = xr.DataArray(
        np.array([1 if 3 <= x else 0 for x in np.linspace(2, 7, 750)]),
        dims=["x"],
        coords=dict(x=np.linspace(2, 7, 750)),
    )

    assert ops.Hellinger_distance(Uniform1, Uniform2) == pytest.approx(1.5, abs=1e-2)
