import itertools
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr

from utopya.eval import is_operation

# --- Custom DAG operations for the NeuralABM model --------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# DECORATOR
# ----------------------------------------------------------------------------------------------------------------------


def apply_along_dim(func):
    def _apply_along_axes(
        *args,
        along_dim: list = None,
        exclude_dim: list = None,
        **kwargs,
    ):
        """Decorator which allows for applying a function, acting on aligned array-likes, along dimensions of
        the xarray.Datasets. The datasets must be aligned. All functions using this header should therefore only take
        xarray objects as arguments that can be indexed along common dimensions. All other arguments should be keywords.

        :param args: xr.Datasets which are to be aligned
        :param along_dim: the dimensions along with to apply the operation
        :param exclude_dim: the dimensions to exclude. This is an alternative to providing the 'along_dim' argument.
            Cannot provide both 'along_dim' and 'exclude_dim'
        :param kwargs: passed to function
        :return: xr.Dataset
        """
        if along_dim and exclude_dim:
            raise ValueError("Cannot provide both 'along_dim' and 'exclude_dim'!")

        if along_dim is not None or exclude_dim is not None:
            # Get the coordinates for all the dimensions that are to be excluded
            if exclude_dim is None:
                excluded_dims = []
                for c in list(args[0].coords.keys()):
                    if c not in along_dim:
                        excluded_dims.append(c)
            else:
                excluded_dims = exclude_dim
            excluded_coords = [args[0].coords[_].data for _ in excluded_dims]

            # Collect the dsets into one dataset
            dsets = []

            # Iterate over all coordinates in the dimensions and apply the function separately
            for idx in itertools.product(*(range(len(_)) for _ in excluded_coords)):
                # Strip both datasets of all coords except the ones along which the function is being
                # applied. Add the coordinates back afterwards and re-merge.
                dsets.append(
                    func(
                        *[
                            arg.sel(
                                {
                                    excluded_dims[j]: excluded_coords[j][idx[j]]
                                    for j in range(len(excluded_dims))
                                },
                                drop=True,
                            )
                            for arg in args
                        ],
                        **kwargs,
                    ).expand_dims(
                        dim={
                            excluded_dims[i]: [excluded_coords[i][idx[i]]]
                            for i in range(len(excluded_dims))
                        }
                    )
                )

            # Merge the datasets into one and return
            return xr.merge(dsets)

        else:
            return func(*args, **kwargs)

    return _apply_along_axes


# ----------------------------------------------------------------------------------------------------------------------
# DATA RESHAPING AND REORGANIZING
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("concat_along")
def concat(objs: list, name: str, dims: Sequence, *args, **kwargs):
    """Combines the pd.Index and xr.concat functions into one.

    :param objs: the xarrays to be concatenated
    :param name: the name of the new dimension
    :param dims: the coordinates of the new dimension
    :param args: passed to xr.concat
    :param kwargs: passed to xr.concat
    :return:
    """
    return xr.concat(objs, pd.Index(dims, name=name), *args, **kwargs)


@is_operation("flatten_dims")
@apply_along_dim
def flatten_dims(
    ds: xr.Dataset,
    *,
    dims: dict,
    new_coords: list = None,
) -> xr.Dataset:
    """Flattens dimensions of an xr.Dataset into a new dimension. New coordinates can be assigned,
    else the dimension is simply given trivial dimensions. The operation is a combination of stacking and
    subsequently dropping the multiindex.

    :param ds: the xr.Dataset to reshape
    :param dims: a dictionary, keyed by the name of the new dimension, and with the dimensions to be flattened as the value
    :param new_coords: the coordinates for the new dimension (optional)
    """
    new_dim, dims_to_stack = list(dims.keys())[0], list(dims.values())[0]

    # Check if the new dimension name already exists. If it already exists, use a temporary name for the new dimension
    # switch back later
    _renamed = False
    if new_dim in list(ds.coords.keys()):
        new_dim = f"__{new_dim}__"
        _renamed = True

    # Stack and drop the dimensions
    ds = ds.stack({new_dim: dims_to_stack})
    q = set(dims_to_stack)
    q.add(new_dim)
    ds = ds.drop_vars(q)

    # Name the stacked dimension back to the originally intended name
    if _renamed:
        ds = ds.rename({new_dim: list(dims.keys())[0]})
        new_dim = list(dims.keys())[0]
    # Add coordinates to new dimension and return
    if new_coords is None:
        return ds.assign_coords({new_dim: np.arange(len(ds.coords[new_dim]))})
    else:
        return ds.assign_coords({new_dim: new_coords})


@is_operation("broadcast")
@apply_along_dim
def broadcast(
    ds1: Union[xr.DataArray, xr.Dataset],
    ds2: Union[xr.DataArray, xr.Dataset],
    *,
    x: str = "x",
    p: str = "loss",
) -> xr.Dataset:
    """Broadcasts together two datasets and returns a dataset with the variables of the datasets set as dimensions.
    This is typically used to broadcast together a dataset of parameters of dimension `(D, N)` and a dataset of associated
    loss values of dimension `(N, )` into a new dataset of dimension `(D, 2, N)`, where each parameter now has the loss
    value associated with it, allowing for easy computation of marginals etc.

    :param ds1: the first dataset
    :param ds2: the second dataset
    :param x: name for the new first dimension of the new variable
    :param p: name for the new second dimension of the new variable
    :return: ``xr.Dataset`` with the previous variables now dimensions of the dataset
    """
    return xr.broadcast(xr.Dataset({x: ds1, p: ds2}))[0]


# ----------------------------------------------------------------------------------------------------------------------
# BASIC STATISTICS FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("stat")
@apply_along_dim
def stat_function(
    data: xr.Dataset, *, stat: str, x: str = None, p: str = None
) -> xr.Dataset:
    """Basic statistical function which returns statistical properties of a one-dimensional dataset representing
    x and f(x)-values.

    :param data: dataset along which to calculate the statistic
    :param stat: type of statistic to calculate
    :param x: label of the x-values; can be a variable in the dataset or a coordinate
    :param p: function values
    :return: the computed statistic
    """
    if x is None:
        x = list(data.coords.keys())[0]
    if p is None:
        p = list(data.data_vars.keys())[0]
    if x in data.coords.keys():
        _x_vals = data.coords[x]
    else:
        _x_vals = data[x]

    _x_vals, _y_vals = _x_vals[~np.isnan(data[p])], data[p][~np.isnan(data[p])]

    # Expectation value
    if stat == "mean":
        _res = scipy.integrate.trapezoid(_y_vals * _x_vals, _x_vals)
    # Standard deviation
    elif stat == "std":
        _m = mean(data, x=x, p=p).to_array().data
        _res = np.sqrt(
            scipy.integrate.trapezoid(_y_vals * (_x_vals - _m) ** 2, _x_vals)
        )

    # Interquartile range
    elif stat == "iqr":
        _int = scipy.integrate.trapezoid(_y_vals, _x_vals)
        _a_0 = -1.0
        __int = 0.0
        _res = 0.0
        for i in range(1, len(_x_vals)):
            __int += scipy.integrate.trapezoid(
                _y_vals[i - 1 : i + 1], _x_vals[i - 1 : i + 1]
            )
            if __int > 0.25 * _int and _a_0 == -1:
                _a_0 = _x_vals[i].item()
            if __int > 0.75 * _int:
                _res = _x_vals[i].item() - _a_0
                break
    else:
        raise ValueError(f"Unrecognized statistic '{stat}'!")

    return xr.Dataset(data_vars=dict({stat: _res}))


@is_operation("mean")
@apply_along_dim
def mean(*args, **kwargs) -> xr.Dataset:
    """Computes the mean of a dataset"""
    return stat_function(*args, stat="mean", **kwargs)


@is_operation("std")
@apply_along_dim
def std(*args, **kwargs) -> xr.Dataset:
    """Computes the standard deviation of a dataset"""
    return stat_function(*args, stat="std", **kwargs)


@is_operation("iqr")
@apply_along_dim
def iqr(*args, **kwargs) -> xr.Dataset:
    """Computes the interquartile range of a dataset"""
    return stat_function(*args, stat="iqr", **kwargs)


@is_operation("mode")
@apply_along_dim
def mode(data: xr.Dataset, *, p: str, get: str = "coord", **kwargs) -> xr.Dataset:
    """Returns the x value and associated probability of the mode of a one-dimensional dataset
    consisting of x-values and associated probabilities, (x, p(x)).

    :param data: the one-dimensional dataset
    :param p: the name of the probability dimension along which to select the mode.
    :param get: whether to return the `value` or the `coord` of the mode. Default is the coordinate, i.e. the x-value
    :param kwargs: keyword arguments passed to xr.DataArray.idxmax
    :return: the mode of the dataset
    """

    # Get the name of the dimension
    coord = list(data.dims.keys())[0]

    # Get the index of the mode and select it
    x_max = data[p].idxmax(**kwargs).item()
    mode = data.sel({coord: x_max})
    if get == "coord":
        mode[p] = x_max
    elif get != "value":
        raise ValueError(
            f"Unrecognised argument {get} for get! Can be either 'value' or 'coord'!"
        )
    return mode.drop_vars(coord)


@is_operation("average_peak_widths")
@apply_along_dim
def avg_peak_width(
    data: xr.DataArray, *, pad: bool = True, width=[None, None], **kwargs
) -> xr.Dataset:
    """Computes the average peak width of a one-dimensional DataArray using the scipy.signal.peaks function. Zeros are
    optionally added to the start and end of the dataset to ensure that peaks at the ends of the intervals are found.

    :param data: the dataset
    :param pad: whether to pad the array with zeros at the beginning and end (True by default)
    :param width: the boundaries for the peak widths
    :param kwargs: (optional) additional kwargs, passed to scipy.signal.find_peaks
    :return: xr.Dataset of mean peak width and standard deviation
    """

    # Insert a zero at the beginning and the end of the array
    arr = np.insert(np.insert(data.data, 0, 0), 0, -1) if pad else data.data

    # Find the peaks along the array
    peaks = scipy.signal.find_peaks(arr, width=width, **kwargs)

    # Calculate the mean and standard deviation of the peaks
    mean, std = np.mean(peaks[1]["widths"]), np.std(peaks[1]["widths"])

    return xr.Dataset(data_vars=dict(mean=mean, std=std))


@is_operation("p_value")
@apply_along_dim
def p_value(data: xr.Dataset, *, t: float, x: str, p: str) -> xr.Dataset:
    """Calculates the p value for a one-dimensional point from a Dataset containing pairs (x, p(x)).
    Since the probabilities are binned, they are normalised such that sum p(x)dx = 1.

    :param data: the one-dimensional DataArray containing the x and p(x) values
    :param t: the value (1d point) for which to calculate the p-value
    :param x: the x value dimension
    :param p: the probability value dimension
    :return: an xr.Dataset of the p-value
    """

    # Calculate the mean
    mu = mean(data, x=x, p=p).to_array().data

    # Calculate the index of t
    t_index = np.argmin(np.abs(data[x] - t).data)

    # Calculate the p-value depending on the location of t
    if t >= mu:
        return xr.Dataset(
            data_vars=dict(
                p_value=scipy.integrate.trapezoid(
                    data[p][t_index:], data[x].values[t_index:]
                )
            )
        )
    else:
        return xr.Dataset(
            data_vars=dict(
                p_value=scipy.integrate.trapezoid(
                    data[p][:t_index], data[x].values[:t_index]
                )
            )
        )


# ----------------------------------------------------------------------------------------------------------------------
# HISTOGRAMS
# ----------------------------------------------------------------------------------------------------------------------


def _hist(obj, *, normalize, **_kwargs):
    # Applies numpy histogram along an axis and returns the counts and bin centres
    _counts, _edges = np.histogram(obj, **_kwargs)
    _counts = _counts.astype(float)
    _bin_centres = np.round((_edges[:-1] + _edges[1:]) / 2, 2)
    if normalize:
        norm = scipy.integrate.trapezoid(_counts, _bin_centres)
        norm = 1.0 if norm == 0.0 else norm
        _counts /= norm if isinstance(normalize, bool) else normalize / norm
    return _counts, _bin_centres


def _get_hist_bins_ranges(ds, bins, ranges, axis):
    # Get the bins and range objects
    if isinstance(bins, xr.DataArray):
        bins = bins.data

    if ranges is not None:
        ranges = (
            np.array(ranges.data)
            if isinstance(ranges, xr.DataArray)
            else np.array(ranges)
        )
        for idx in range(len(ranges)):
            if ranges[idx] is None:
                ranges[idx] = (
                    np.min(ds.data[axis]) if idx == 0 else np.max(ds.data[axis])
                )
    return bins, ranges


@is_operation("hist")
@apply_along_dim
def hist(
    ds: Union[xr.DataArray, xr.Dataset],
    bins: Any = 100,
    ranges: Any = None,
    *,
    axis: int = -1,
    normalize: Union[float, bool] = False,
    **kwargs,
) -> xr.Dataset:
    """Applies `np.histogram` using the apply_along_dim decorator to allow histogramming along multiple
    dimensions. If binning is only desired along a single dimension, `hist_1D` is significantly faster
    since splitting and merging operations is not required.

    :param ds: the DataArray on which to apply the histogram function
    :param bins: the bins to use, passed to `np.histogram`. This can be a single integer, in which case it is
        interpreted as the number of bins, a Sequence defining the bin edges, or a string defining the method to use.
        See `np.histogram` for details
    :param ranges: (float, float), optional: the lower and upper range of the bins
    :param axis: the axis along which to apply np.histogram. By default, this happens on the innermost axis.
    :param normalize: whether to normalize the counts. Can be a boolean or a float, in which case the counts are
        normalized to that value
    :param kwargs: passed to `np.histogram`
    """

    if isinstance(ds, xr.Dataset):
        ds = ds.to_array().squeeze()

    # Get the bins and range objects
    bins, ranges = _get_hist_bins_ranges(ds, bins, ranges, axis)

    # Get the name of the dimension
    dim = ds.name if ds.name is not None else "_variable"

    # Apply the histogram function along the axis
    counts, bin_centres = np.apply_along_axis(
        _hist, axis, ds, bins=bins, range=ranges, normalize=normalize, **kwargs
    )

    return xr.Dataset(
        data_vars={dim: ("bin_idx", counts), "x": ("bin_idx", bin_centres)},
        coords={"bin_idx": np.arange(len(bin_centres))},
    )


@is_operation("hist_1D")
def hist_1D(
    ds: Union[xr.DataArray, xr.Dataset],
    bins: Any = 100,
    ranges: Any = None,
    *,
    along_dim: Sequence = None,
    axis: int = None,
    normalize: Union[bool, float] = False,
    **kwargs,
) -> xr.Dataset:
    """Applies ``np.histogram`` along a single axis of a dataset. This bypasses the `apply_along_axis` decorator
    and is therefore significantly faster than ``hist``."""

    if along_dim is None and axis is None:
        raise ValueError("One of either 'along_dim' or 'axis' must be passed!")
    if along_dim is not None:
        if len(along_dim) > 1:
            raise ValueError(
                "Cannot use the `hist_1D` function for multidimensional histogram operations!"
                "Use `hist` instead."
            )
        along_dim = along_dim[0]
        axis = list(ds.dims).index(along_dim)
    else:
        # Get the axis of the dimension along which the operation is to be applied
        along_dim = list(ds.coords.keys())[axis]

    # Get the name of the dimension
    dim = ds.name if ds.name is not None else "_variable"

    # Get the histogram bins and ranges
    bins, ranges = _get_hist_bins_ranges(ds, bins, ranges, axis)

    # Apply the histogram function along the axis
    res = np.apply_along_axis(
        _hist, axis, ds, bins=bins, range=ranges, normalize=normalize, **kwargs
    )

    # Get the counts and the bin centres. Note that the bin centres are equal along every dimension!
    counts, bin_centres = np.take(res, 0, axis=axis), np.take(res, 1, axis=axis)
    sel = [0] * len(np.shape(bin_centres))
    sel[axis] = None
    bin_centres = bin_centres[*sel].flatten()

    # Put the dataset back together again, relabelling the coordinate dimension that was binned
    coords = dict(ds.coords)
    coords.update({along_dim: bin_centres})

    res = xr.Dataset(data_vars={dim: (list(ds.sizes.keys()), counts)}, coords=coords)
    return res.rename({along_dim: "x"})


# ----------------------------------------------------------------------------------------------------------------------
# PROBABILITY DENSITY OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


@is_operation("joint_2D")
@apply_along_dim
def joint_2D(
    x: xr.DataArray,
    y: xr.DataArray,
    values: xr.DataArray,
    bins: Union[int, xr.DataArray] = 100,
    ranges: xr.DataArray = None,
    *,
    statistic: Union[str, callable] = "mean",
    normalize: Union[bool, float] = False,
    differential: float = None,
    dim_names: Sequence = ("x", "p"),
    **kwargs,
) -> xr.DataArray:
    """
    Computes the two-dimensional joint distribution of a dataset of parameters by calling the scipy.stats.binned_statistic_2d
    function.

    The function returns a statistic for each bin (typically the mean).

    :param x: DataArray of values in the first dimension
    :param y: DataArray of values in the second dimension
    :param values: DataArray of values to be binned
    :param bins: bins argument to `scipy.binned_statistic_2d`
    :param ranges: range arguent to `scipy.binned_statistic_2d`
    :param normalize: whether to normalize the joint (False by default), and the normalisation value (1 by default)
    :param differential: the spacial differential dx to use for normalisation. Defaults to the grid spacing
    :param dim_names (optional): names of the two dimensions
    :return: an xr.Dataset of the joint distribution
    """

    # Get the number of bins
    if isinstance(bins, xr.DataArray):
        bins = bins.data

    # Allow passing 'None' arguments in the plot config for certain entries of the range arg
    # This allows clipping only on some dimensions without having to specify every limit
    if ranges is not None:
        ranges = np.array(ranges.data)
        for idx in range(len(ranges)):
            if None in ranges[idx]:
                ranges[idx] = (
                    [np.min(x), np.max(x)] if idx == 0 else [np.min(y), np.max(y)]
                )

    # Get the statistics and bin edges
    stat, x_edge, y_edge, _ = scipy.stats.binned_statistic_2d(
        x, y, values, statistic=statistic, bins=bins, range=ranges, **kwargs
    )
    # Normalise the joint distribution, if given
    if normalize:
        dxdy = (
            np.prod([a[1] - a[0] for a in [x_edge, y_edge]])
            if differential is None
            else differential
        )
        norm = np.nansum(stat) * dxdy
        if norm == 0:
            norm = 1
        stat /= norm if isinstance(normalize, bool) else norm / normalize

    return xr.DataArray(
        data=stat,
        dims=dim_names,
        coords={
            dim_names[0]: 0.5 * (x_edge[1:] + x_edge[:-1]),
            dim_names[1]: 0.5 * (y_edge[1:] + y_edge[:-1]),
        },
        name="joint",
    )


@is_operation("joint_2D_ds")
@apply_along_dim
def joint_2D_ds(
    ds: Union[xr.DataArray, xr.Dataset],
    values: xr.DataArray,
    bins: xr.DataArray = 100,
    ranges: xr.DataArray = None,
    *,
    x: str,
    y: str,
    **kwargs,
) -> xr.DataArray:
    """Computes a two-dimensional joint from a single dataset with x and y given as variables, or from
    a DataArray with x and y given as coordinate dimensions."""

    if isinstance(ds, xr.Dataset):
        return joint_2D(ds[x], ds[y], values, bins, ranges, dim_names=(x, y), **kwargs)
    elif isinstance(ds, xr.DataArray):
        return joint_2D(
            ds.sel(dict(parameter=x)),
            ds.sel(dict(parameter=y)),
            values,
            bins,
            ranges,
            dim_names=(x, y),
            **kwargs,
        )


@is_operation("marginal_from_joint")
@apply_along_dim
def marginal_from_joint(
    joint: Union[xr.DataArray, xr.Dataset],
    *,
    parameter: str,
    normalize: Union[bool, float] = True,
    scale_y_bins: bool = False,
) -> xr.Dataset:
    """
    Computes a marginal from a two-dimensional joint distribution by summing over one parameter. Normalizes
    the marginal, if specified. NaN values in the joint are skipped when normalising: they are not zero, just unknown.
    Since x-values may differ for different parameters, the x-values are variables in a dataset, not coordinates.
    The coordinates are given by the bin index, thereby allowing marginals across multiple parameters to be combined
    into a single xr.Dataset.

    :param joint: the joint distribution over which to marginalise
    :param normalize: whether to normalize the marginal distribution. If true, normalizes to 1, else normalizes to
        a given value
    :param scale_y_bins: whether to scale the integration over y by range of the given values (y_max - y_min)
    """

    # Get the integration coordinate
    integration_coord = [c for c in list(joint.coords) if c != parameter][0]

    # Marginalise over the integration coordinate
    marginal = []
    for p in joint.coords[parameter]:
        _y, _x = joint.sel({parameter: p}).data, joint.coords[integration_coord]
        if scale_y_bins and not np.isnan(_y).all():
            _f = np.nanmax(_y) - np.nanmin(_y)
            _f = 1.0 / _f if _f != 0 else 1.0
        else:
            _f = 1.0
        marginal.append(
            _f * scipy.integrate.trapezoid(_y[~np.isnan(_y)], _x[~np.isnan(_y)])
        )

    # Normalise, if given
    if normalize:
        norm = scipy.integrate.trapezoid(marginal, joint.coords[parameter])
        if norm == 0:
            norm = 1
        marginal /= norm if isinstance(normalize, bool) else norm / normalize

    # Return a dataset with x- and y-values as variables, and coordinates given by the bin index
    # This allows combining different marginals with different x-values but identical number of bins
    # into a single dataset
    return xr.Dataset(
        data_vars=dict(
            x=(["bin_idx"], joint.coords[parameter].data),
            marginal=(["bin_idx"], marginal),
        ),
        coords=dict(
            bin_idx=(["bin_idx"], np.arange(len(joint.coords[parameter].data)))
        ),
    )


@is_operation("marginal")
@apply_along_dim
def marginal(
    x: xr.DataArray,
    prob: xr.DataArray,
    bins: Union[int, xr.DataArray],
    ranges: Union[Sequence, xr.DataArray],
    *,
    parameter: str = "x",
    normalize: Union[bool, float] = True,
    scale_y_bins: bool = False,
    **kwargs,
) -> xr.Dataset:
    """
    Computes a marginal directly from a xr.DataArray of x-values and a xr.DataArray of probabilities by first
    computing the joint distribution and then marginalising over the probability. This way, points that are sampled
    multiple times only contribute once to the marginal, which is not a representation of the frequency with which
    each point is sampled, but of the calculated likelihood function.

    :param x: array of samples in the first dimension (the parameter estimates)
    :param prob: array of samples in the second dimension (the unnormalised probability value)
    :param bins: bins to use for both dimensions
    :param range: range to use for both dimensions. Defaults to the minimum and maximum along each dimension
    :param parameter: the parameter over which to marginalise. Defaults to the first dimension.
    :param normalize: whether to normalize the marginal
    :param scale_y_bins: whether to scale the integration over y by range of the given values (y_max - y_min)
    :param kwargs: other kwargs, passed to the joint_2D function
    :return: an xr.Dataset of the marginal densities
    """
    joint = joint_2D(x, prob, prob, bins, ranges, normalize=normalize, **kwargs)
    return marginal_from_joint(
        joint, parameter=parameter, normalize=normalize, scale_y_bins=scale_y_bins
    )


@is_operation("marginal_from_ds")
@apply_along_dim
def marginal_from_ds(
    ds: xr.Dataset,
    bins: xr.DataArray = 100,
    ranges: xr.DataArray = None,
    *,
    x: str,
    y: str,
    **kwargs,
) -> xr.Dataset:
    """Computes the marginal from a single dataset with x and y given as variables."""
    return marginal(ds[x], ds[y], bins, ranges, **kwargs)


@is_operation("joint_DD")
@apply_along_dim
def joint_DD(
    sample: xr.DataArray,
    values: xr.DataArray,
    bins: Union[int, xr.DataArray] = 100,
    ranges: xr.DataArray = None,
    *,
    statistic: Union[str, callable] = "mean",
    normalize: Union[bool, float] = False,
    differential: float = None,
    dim_names: Sequence,
    **kwargs,
) -> xr.DataArray:
    """
    Computes the d-dimensional joint distribution of a dataset of parameters by calling the scipy.stats.binned_statistic_dd
    function. This function can handle at most 32 parameters.

    The function returns a statistic for each bin (typically the mean).

    :param sample: DataArray of values
    :param values: DataArray of values to be binned
    :param bins: bins argument to `scipy.binned_statistic_dd`
    :param ranges: range arguent to `scipy.binned_statistic_dd`
    :param normalize: whether to normalize the joint (False by default), and the normalisation value (1 by default)
    :param differential: the spacial differential dx to use for normalisation. Defaults to the grid spacing
    :param dim_names (optional): names of the two dimensions
    :return: an xr.Dataset of the joint distribution
    """

    # Get the number of bins
    if isinstance(bins, xr.DataArray):
        bins = bins.data

    # Allow passing 'None' arguments in the plot config for certain entries of the range arg
    # This allows clipping only on some dimensions without having to specify every limit
    if ranges is not None:
        ranges = ranges.data
        for idx in range(len(ranges)):
            if None in ranges[idx]:
                ranges[idx] = [np.min(sample.coords[idx]), np.max(sample.coords[idx])]

    # Get the statistics and bin edges
    stat, bin_edges, _ = scipy.stats.binned_statistic_dd(
        sample, values, statistic=statistic, bins=bins, range=ranges, **kwargs
    )
    # Normalise the joint distribution, if given
    if normalize:
        differential = (
            np.prod([a[1] - a[0] for a in bin_edges])
            if differential is None
            else differential
        )
        norm = np.nansum(stat) * differential
        if norm == 0:
            norm = 1
        stat /= norm if isinstance(normalize, bool) else norm / normalize

    return xr.DataArray(
        data=stat,
        dims=dim_names,
        coords={
            dim_names[i]: 0.5 * bin_edges[i][1:] + bin_edges[i][:-1]
            for i in range(len(bin_edges))
        },
        name="joint",
    )


def _interpolate(_p, _q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects two densities onto a common grid"""
    _dim1, _dim2 = list(_p.coords.keys())[0], list(_q.coords.keys())[0]

    # Return densities if they are already equal
    if all(_p.coords[_dim1].data == _q.coords[_dim2].data):
        return _p, _q, _p.coords[_dim1].data

    # Generate a common grid
    _x_min, _x_max = np.max(
        [_p.coords[_dim1][0].item(), _q.coords[_dim2][0].item()]
    ), np.min([_p.coords[_dim1][-1].item(), _q.coords[_dim2][-1].item()])

    # Interpolate the functions onto a common grid
    _grid = np.linspace(_x_min, _x_max, len(_p.coords[_dim1]) + len(_q.coords[_dim2]))
    _p_interp = np.interp(_grid, _p.coords[_dim1], _p)
    _q_interp = np.interp(_grid, _q.coords[_dim2], _q)

    return _p_interp, _q_interp, _grid


@is_operation("Hellinger_distance")
@apply_along_dim
def Hellinger_distance(
    p: xr.DataArray,
    q: xr.DataArray,
) -> xr.Dataset:
    r"""Calculates the Hellinger distance between two distributions p and q, defined as

        d_H(p, q) = 1/2 * \int sqrt(p(x)) - sqrt(q(x))**2 dx.

    The Hellinger distance is calculated on the common support of p and q; if p and q have different discretisation
    levels, the functions are interpolated onto a common grid.

    :param p: one-dimensional arrays of density values for p
    :param q: one-dimensional arrays of density values for q
    :return: the Hellinger distance between p and q
    """

    p_interp, q_interp, grid = _interpolate(p, q)

    # Calculate the Hellinger distance and return
    return xr.Dataset(
        data_vars=dict(
            Hellinger_distance=0.5
            * scipy.integrate.trapezoid(
                np.square(np.sqrt(p_interp) - np.sqrt(q_interp)), grid
            )
        )
    )


@is_operation("relative_entropy")
@apply_along_dim
def relative_entropy(
    p: Union[xr.Dataset, xr.DataArray], q: Union[xr.Dataset, xr.DataArray]
) -> xr.Dataset:
    """Calculates the relative_entropy distance between two distributions p and q, defined as

        d_KL(p, q) = int p(x) log(p(x)/q(x))dx

    Canonically, log(0/0) = log(1/1) = 0.
    The relative entropy (KL divergence) is calculated on the common support of p and q; if p and q have different
    discretisation levels, the functions are interpolated onto a common grid.

    :param p: one-dimensional arrays of density values for p
    :param q: one-dimensional arrays of density values for q
    :return: the relative entropy between p and q

    """
    p_interp, q_interp, grid = _interpolate(p, q)

    p_interp = np.where(p_interp != 0, p_interp, 1.0)
    q_interp = np.where(q_interp != 0, q_interp, 1.0)
    return xr.Dataset(
        data_vars=dict(
            relative_entropy=scipy.integrate.trapezoid(
                p_interp * np.log(p_interp / q_interp), grid
            )
        )
    )


@is_operation("L2_distance")
@apply_along_dim
def L2_distance(p: xr.DataArray, q: xr.DataArray) -> xr.Dataset:
    """Calculates the L2 distance between two distributions p and q, defined as

        d_H(p, q) = sqrt(int (p(x) - q(x))**2 dx)

    :param p: one-dimensional array of density values
    :param q: one-dimensional array of density values
    :return: the L2 distance between p and q
    """
    return xr.Dataset(data_vars=dict(std=np.sqrt(np.square(p - q).sum())))


# ----------------------------------------------------------------------------------------------------------------------
# MCMC operations
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("batch_mean")
@apply_along_dim
def batch_mean(da: xr.DataArray, *, batch_size: int = None) -> xr.Dataset:
    """Computes the mean of a single sampling chain over batches of length B. Default batch length is
    int(sqrt(N)), where N is the length of the chain.

    :param da: dataarray of samples
    :param batch_size: batch length over which to compute averages
    :return: res: averages of the batches
    """
    vals = da.data
    means = np.array([])
    windows = np.arange(0, len(vals), batch_size)
    if len(windows) == 1:
        windows = np.append(windows, len(vals) - 1)
    else:
        if windows[-1] != len(vals) - 1:
            windows = np.append(windows, len(vals) - 1)
    for idx, start_idx in enumerate(windows[:-1]):
        means = np.append(means, np.mean(vals[start_idx : windows[idx + 1]]))

    return xr.Dataset(
        data_vars=dict(means=("batch_idx", means)),
        coords=dict(batch_idx=("batch_idx", np.arange(len(means)))),
    )


@is_operation("gelman_rubin")
@apply_along_dim
def gelman_rubin(da: xr.Dataset, *, step_size: int = 1) -> xr.Dataset:
    R = []
    for i in range(step_size, len(da.coords["sample"]), step_size):
        da_sub = da.isel({"sample": slice(0, i)})
        L = len(da_sub.coords["sample"])

        chain_mean = da_sub.mean("sample")
        between_chain_variance = L * chain_mean.std("seed", ddof=1) ** 2
        within_chain_variance = da_sub.std("sample", ddof=1) ** 2
        W = within_chain_variance.mean("seed")
        R.append(((L - 1) * W / L + 1 / L * between_chain_variance) / W)

    return xr.Dataset(
        data_vars=dict(gelman_rubin=("sample", R)),
        coords=dict(
            sample=("sample", np.arange(step_size, len(da.coords["sample"]), step_size))
        ),
    )


# ----------------------------------------------------------------------------------------------------------------------
# CSV operations
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("to_csv")
def to_csv(
    data: Union[xr.Dataset, xr.DataArray], path: str
) -> Union[xr.Dataset, xr.DataArray]:
    df = data.to_dataframe()
    df.to_csv(path)
    return data
