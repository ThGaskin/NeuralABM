from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr

from utopya.eval import is_operation

from ._op_utils import _get_hist_bins_ranges, _hist, _interpolate, apply_along_dim

# --- Custom DAG operations for the NeuralABM model --------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# DATA RESHAPING AND REORGANIZING
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("concat_along")
def concat(objs: Sequence, name: str, dims: Sequence, *args, **kwargs):
    """Combines the pd.Index and xr.concat functions into one.

    :param objs: the xarray objects to be concatenated
    :param name: the name of the new dimension
    :param dims: the coordinates of the new dimension
    :param args: passed to ``xr.concat``
    :param kwargs: passed to ``xr.concat``
    :return: objects concatenated along the new dimension
    """
    return xr.concat(objs, pd.Index(dims, name=name), *args, **kwargs)


@is_operation("flatten_dims")
@apply_along_dim
def flatten_dims(
    ds: Union[xr.Dataset, xr.DataArray],
    *,
    dims: dict,
    new_coords: Sequence = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Flattens dimensions of an xarray object into a new dimension. New coordinates can be assigned,
    else the dimension is simply given trivial dimensions. The operation is a combination of stacking and
    subsequently dropping the multiindex.

    :param ds: the xarray object to reshape
    :param dims: a dictionary, keyed by the name of the new dimension, and with the dimensions to be flattened as the value
    :param new_coords: (optional) coordinates for the new dimension
    :return the xarray object with flattened dimensions
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
    ds1: xr.DataArray, ds2: xr.DataArray, *, x: str = "x", p: str = "loss", **kwargs
) -> xr.Dataset:
    """Broadcasts together two ``xr.DataArray`` s and returns a dataset with given ``x`` and ``p`` as variable names.

    :param ds1: the first array
    :param ds2: the second array
    :param x: name for the new first variable
    :param p: name for the new second variable
    :param kwargs: passed on to ``xr.broadcast``
    :return: ``xr.Dataset`` with variables ``x`` and ``p``
    """
    return xr.broadcast(xr.Dataset({x: ds1, p: ds2}), **kwargs)[0]


# ----------------------------------------------------------------------------------------------------------------------
# BASIC STATISTICS FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("stat")
@apply_along_dim
def stat_function(
    data: xr.Dataset, *, stat: str, x: str, y: str, **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """Basic statistical function which returns statistical properties of a one-dimensional dataset representing
    x and y(x)-values.

    :param data: ``xr.Dataset`` along which to calculate the statistic. The dataset must contain the ``y`` key as
        a variable, but ``x`` may also be a coordinate name.
    :param stat: type of statistic to calculate: can be ``mean``, ``std``, ``iqr``, ``mode``, or ``avg_peak_width``.
        When calculating the mode, both the x-value and y-value are returned, and when calculating peak widths, both the
        mean width and standard deviation are calculated.
    :param x: label of the x-values; can be a variable in the dataset or a coordinate
    :param y: function values
    :param kwargs: kwargs passed to the respective calculation function
    :return: the computed statistic
    """

    _permitted_stat_functions = ["mean", "std", "iqr", "mode", "avg_peak_width"]
    if stat not in _permitted_stat_functions:
        raise ValueError(
            f"Unrecognised stat function '{stat}'; choose from '{', '.join(_permitted_stat_functions)}'."
        )

    # x-values can be either a variable or a coordinate
    if x in data.coords.keys():
        _x_vals = data.coords[x]
    else:
        _x_vals = data[x]

    # Ignore nans in the values
    _x_vals, _y_vals = _x_vals[~np.isnan(data[y])], data[y][~np.isnan(data[y])]

    # ------------------------------------------------------------------------------------------------------------------
    # Expectation value: m = int f(x) x dx
    # ------------------------------------------------------------------------------------------------------------------

    if stat == "mean":
        _res = scipy.integrate.trapezoid(_y_vals * _x_vals, _x_vals, **kwargs)
        return xr.DataArray(_res, name=stat)

    # ------------------------------------------------------------------------------------------------------------------
    # Standard deviation: std^2 = int (x - m)^2 f(x) dx
    # ------------------------------------------------------------------------------------------------------------------

    elif stat == "std":
        _m = stat_function(data, x=x, y=y, stat="mean")
        _res = np.sqrt(
            scipy.integrate.trapezoid(_y_vals * (_x_vals - _m) ** 2, _x_vals, **kwargs)
        )
        return xr.DataArray(_res, name=stat)

    # ------------------------------------------------------------------------------------------------------------------
    # Interquartile range: length between first and third quartile
    # ------------------------------------------------------------------------------------------------------------------

    elif stat == "iqr":
        _int = scipy.integrate.trapezoid(_y_vals, _x_vals, **kwargs)
        _a_0 = -1.0
        __int = 0.0
        _res = 0.0
        for i in range(1, len(_x_vals)):
            __int += scipy.integrate.trapezoid(
                _y_vals[i - 1 : i + 1], _x_vals[i - 1 : i + 1], **kwargs
            )
            if __int > 0.25 * _int and _a_0 == -1:
                _a_0 = _x_vals[i].item()
            if __int > 0.75 * _int:
                _res = _x_vals[i].item() - _a_0
                break
        return xr.DataArray(_res, name=stat)

    # ------------------------------------------------------------------------------------------------------------------
    # Mode: both the x-value and the y-value of the mode are returned
    # ------------------------------------------------------------------------------------------------------------------

    elif stat == "mode":
        # Get the index of the mode and select it
        idx_max = np.argmax(_y_vals.data, **kwargs)
        mode_x = _x_vals[idx_max]
        mode_y = _y_vals[idx_max]
        return xr.Dataset(data_vars=dict(mode_x=mode_x, mode_y=mode_y))

    # ------------------------------------------------------------------------------------------------------------------
    # Average peak width: both the mean width and standard deviation of the widths is returned
    # ------------------------------------------------------------------------------------------------------------------

    elif stat == "avg_peak_width":
        if "width" not in kwargs:
            raise Exception("'width' kwarg required for 'scipy.signal.find_peaks'!")

        # Insert a zero at the beginning and the end of the array to ensure peaks at the ends are found
        _y_vals = np.insert(np.insert(_y_vals, 0, 0), 0, -1)

        # Find the peaks along the array
        peaks = scipy.signal.find_peaks(_y_vals, **kwargs)

        # Calculate the mean and standard deviation of the peaks
        mean, std = (
            np.mean(peaks[1]["widths"]) * np.diff(_x_vals)[0],
            np.std(peaks[1]["widths"]) * np.diff(_x_vals)[0],
        )

        return xr.Dataset(data_vars=dict(mean_peak_width=mean, peak_width_std=std))


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
def mode(*args, **kwargs) -> xr.Dataset:
    """Computes the mode of a dataset"""
    return stat_function(*args, stat="mode", **kwargs)


@is_operation("avg_peak_width")
@apply_along_dim
def avg_peak_width(*args, **kwargs) -> xr.Dataset:
    """Computes the average peak width and std of peak widths of a dataset"""
    return stat_function(*args, stat="avg_peak_width", **kwargs)


@is_operation("p_value")
@apply_along_dim
def p_value(
    data: xr.Dataset, point: Any, *, x: str, y: str, null: str = "mean"
) -> xr.DataArray:
    """Calculates the p value of a ``point`` from a Dataset containing x-y-pairs. It is assumed the integral under y
    is normalised for the p-value to be meaningful. The p-value can be calculated wrt to the mean or the mode
    of the distribution

    :param data: ``xr.Dataset`` containing the x and p(x) values
    :param point: point at which to calculate the p value
    :param x: label of the x-values; can be a variable in the dataset or a coordinate
    :param y: function values; assumed to be normalised
    :param null: (optional) null wrt which the p-value is to be calculated; can be either ``mean``
        or ``mode``
    :return: ``xr.DataArray`` of the p-value of ``point``
    """

    # x can be both a variable and a coordinate
    if x in data.coords.keys():
        _x_vals = data.coords[x]
    else:
        _x_vals = data[x]

    if isinstance(point, xr.DataArray):
        point = point.data

    # Calculate the value of the null of the distribution
    m = (
        mean(data, x=x, y=y).data
        if null == "mean"
        else mode(data, x=x, y=y)["mode_x"].data
    )

    # Calculate the index of the point
    t_index = np.argmin(np.abs(_x_vals - point).data)

    # Calculate the p-value depending on the location of the point
    if point >= m:
        return xr.DataArray(
            scipy.integrate.trapezoid(data[y][t_index:], _x_vals[t_index:]),
            name="p_value",
        )

    else:
        return xr.DataArray(
            scipy.integrate.trapezoid(data[y][:t_index], _x_vals[:t_index]),
            name="p_value",
        )


@is_operation("normalize")
@apply_along_dim
def normalize(
    distribution: xr.Dataset, *, x: str, y: str, norm: float = 1, **kwargs
) -> xr.Dataset:
    """Normalises a probability distribution of x- and y-values

    :param distribution: ``xr.Dataset`` of x- and y-values
    :param x: the x-values
    :param y: the function values
    :param norm: (optional) value to which to normalise the distribution
    :param kwargs: passed to ``scipy.integrate.trapezoid``
    :return: the normalised probability distribution
    """

    integral = scipy.integrate.trapezoid(distribution[y], distribution[x], **kwargs)
    distribution[y] *= norm / integral
    return distribution


# ----------------------------------------------------------------------------------------------------------------------
# HISTOGRAMS
# ----------------------------------------------------------------------------------------------------------------------


@is_operation("hist")
@apply_along_dim
def hist(
    da: xr.DataArray,
    bins: Any = 100,
    ranges: Any = None,
    *,
    dim: str,
    axis: int = None,
    normalize: Union[float, bool] = False,
    use_bins_as_coords: bool = False,
    **kwargs,
) -> Union[xr.Dataset, xr.DataArray]:
    """Applies ``np.histogram`` using the ``apply_along_dim`` decorator to allow histogramming along multiple
    dimensions. This function applies ``np.histogram`` along a single axis of an ``xr.DataArray`` object;
    it is recommended to only use ``apply_along_dim`` across small dimensions, as splitting and recombining the
    xarray objects is very expensive.

    :param da: the ``xr.DataArray`` on which to apply the histogram function
    :param bins: the bins to use, passed to ``np.histogram``. This can be a single integer, in which case it is
        interpreted as the number of bins, a Sequence defining the bin edges, or a string defining the method to use.
        See ``np.histogram`` for details
    :param ranges: (optional): the lower and upper range of the bins
    :param dim: the dimension along which to apply the operation. If not passed, an ``axis`` argument must be
        provided
    :param axis: (optional) the axis along which to apply np.histogram.
    :param normalize: whether to normalize the counts. Can be a boolean or a float, in which case the counts are
        normalized to that value
    :param use_bins_as_coords: whether to use the bin centres as coordinates of the dataset, or as variables. If true,
        a ``xr.DataArray`` is returned, with the bin centres as coordinates and the counts as the data. This may
        cause incompatibilities with ``apply_along_dim``, since different samples have different bin_centres. For this
        reason, the default behaviour is to return a ``xr.Dataset`` with the bin_centres and counts as variables,
        and ``bin_idx`` as the coordinate. This enables combining different histograms with different bin centres
        (but same number of bins) into a single dataset. If passed, `ranges` must also be passed to ensure
        all histogram bins are identical.
    :param kwargs: passed to ``np.histogram``
    :return ``xr.DataArray`` or ``xr.Dataset`` containing the bin centres either as coordinates or as variables,
        and the counts.
    """
    if dim is None and axis is None:
        raise ValueError("Must supply either 'dim' or 'axis' arguments!")

    if use_bins_as_coords and ranges is None:
        raise ValueError(
            "Setting 'use_bins_as_coords' to 'True' requires passing a 'ranges' argument to "
            "ensure all coordinates are equal"
        )
    # Get the axis along which to apply the operations
    if dim is not None:
        axis = list(da.dims).index(dim)

    # Get the bins and range objects
    bins, ranges = _get_hist_bins_ranges(da, bins, ranges, axis)

    # Apply the histogram function along the axis
    res = np.apply_along_axis(
        _hist, axis, da, bins=bins, range=ranges, normalize=normalize, **kwargs
    )

    # Get the counts and the bin centres. Note that the bin centres are equal along every dimension!
    counts, bin_centres = np.take(res, 0, axis=axis), np.take(res, 1, axis=axis)

    # Put the dataset back together again, relabelling the coordinate dimension that was binned
    coords = dict(da.coords)

    # Bin centres are to be used as coordinates
    if use_bins_as_coords:
        sel = [0] * len(np.shape(bin_centres))
        sel[axis] = None
        bin_centres = bin_centres[tuple(sel)].flatten()
        coords.update({dim: bin_centres})

        res = xr.DataArray(
            counts,
            dims=list(da.sizes.keys()),
            coords=coords,
            name=da.name if da.name else "count",
        )
        return res.rename({dim: "x"})

    else:
        coords.update({dim: np.arange(np.shape(bin_centres)[axis])})
        other_dim = list(coords.keys())
        other_dim.remove(dim)
        attrs = [*other_dim, "bin_idx"] if other_dim else ["bin_idx"]
        coords["bin_idx"] = coords.pop(dim)

        return xr.Dataset(
            data_vars={
                da.name if da.name else "count": (attrs, counts),
                "x": (attrs, bin_centres),
            },
            coords=coords,
        )


# ----------------------------------------------------------------------------------------------------------------------
# DISTANCES BETWEEN PROBABILITY DENSITIES
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("distances_between_distributions")
@apply_along_dim
def distances_between_distributions(
    P: Union[xr.DataArray, xr.Dataset],
    Q: Union[xr.DataArray, xr.Dataset],
    *,
    stat: str,
    p: float = 2,
    x: str = None,
    y: str = None,
    **kwargs,
) -> xr.DataArray:
    """Calculates distances between two distributions P and Q. Possible distances are:

    - Hellinger distance: d(P, Q) = 1/2 * integral sqrt(P(x)) - sqrt(Q(x))**2 dx.
    - Relative entropy: d(P, Q) = integral P(x) log(P(x)/Q(x))dx
    - Lp distance: d(P, Q) =  ( integral (P(x) - Q(x))^p dx)^{1/p}

    These distances are calculated on the common support of P and Q; if P and Q have different discretisation
    levels, the functions are interpolated.

    :param P: one-dimensional ``xr.DataArray`` or ``xr.Dataset`` of values for P. If ``xr.Dataset``, ``x`` and ``y``
        arguments must be passed.
    :param Q: one-dimensional ``xr.DataArray`` or ``xr.Dataset`` of values for Q. If ``xr.Dataset``, ``x`` and ``y``
        arguments must be passed.
    :param stat: which density to function to use
    :param p: p-value for the Lp distance
    :param x: x-values to use if P and Q are ``xr.Datasets``.
    :param y: y-values to use if P and Q are ``xr.Datasets``.
    :param kwargs: kwargs, passed on to ``scipy.integrate.trapezoid``
    :return: the distance between p and q
    """

    _permitted_stat_functions = ["Hellinger", "relative_entropy", "Lp"]
    if stat not in _permitted_stat_functions:
        raise ValueError(
            f"Unrecognised stat function '{stat}'; choose from '{', '.join(_permitted_stat_functions)}'."
        )

    # If P and Q are datasets, convert to DataArrays
    if isinstance(P, xr.Dataset):
        P = xr.DataArray(
            P[y], coords={"x": P[x] if x in list(P.data_vars) else P.coords[x]}
        )
    if isinstance(Q, xr.Dataset):
        Q = xr.DataArray(
            Q[y], coords={"x": Q[x] if x in list(Q.data_vars) else Q.coords[x]}
        )

    # Interpolate P and Q on their common support
    P, Q, grid = _interpolate(P, Q)

    # Hellinger distance
    if stat == "Hellinger":
        return xr.DataArray(
            0.5
            * scipy.integrate.trapezoid(
                np.square(np.sqrt(P) - np.sqrt(Q)), grid, **kwargs
            ),
            name="Hellinger_distance",
        )

    # Relative entropy
    elif stat == "relative_entropy":
        P, Q = np.where(P != 0, P, 1), np.where(Q != 0, Q, 1)
        return xr.DataArray(
            scipy.integrate.trapezoid(P * np.log(P / Q), grid, **kwargs),
            name="relative_entropy",
        )

    # Lp distance
    elif stat == "Lp":
        return xr.DataArray(
            scipy.integrate.trapezoid((P - Q) ** p, grid, **kwargs) ** (1 / p),
            name=f"Lp_distance",
        )


@is_operation("Hellinger_distance")
@apply_along_dim
def Hellinger_distance(*args, **kwargs) -> xr.DataArray:
    return distances_between_distributions(*args, stat="Hellinger", **kwargs)


@is_operation("relative_entropy")
@apply_along_dim
def relative_entropy(*args, **kwargs) -> xr.DataArray:
    return distances_between_distributions(*args, stat="relative_entropy", **kwargs)


@is_operation("Lp_distance")
@apply_along_dim
def Lp_distance(*args, **kwargs) -> xr.DataArray:
    return distances_between_distributions(*args, stat="Lp", **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# PROBABILITY DENSITY FUNCTIONS
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
    dx: float = None,
    dy: float = None,
    dim_names: Sequence = ("x", "y"),
    **kwargs,
) -> xr.DataArray:
    """
    Computes the two-dimensional joint distribution of a dataset of parameters by calling the scipy.stats.binned_statistic_2d
    function. The function returns a statistic for each bin (typically the mean).

    :param x: DataArray of samples in the first dimension
    :param y: DataArray of samples in the second dimension
    :param values: DataArray of values to be binned
    :param bins: (optional) ``bins`` argument to ``scipy.binned_statistic_2d``
    :param ranges: (optional) ``range`` argument to ``scipy.binned_statistic_2d``
    :param statistic: (optional) ``statistic`` argument to ``scipy.binned_statistic_2d``
    :param normalize: (optional) whether to normalize the joint (False by default), and the normalisation value (1 by default)
    :param dx: (optional) the spacial differential dx to use for normalisation. If provided, the norm will not be
        calculated by integrating against the x-values, but rather by assuming the coordinates are spaced ``dx`` apart
    :param dy: (optional) the spacial differential dy to use for normalisation. If provided, the norm will not be
        calculated by integrating against the y-values, but rather by assuming the coordinates are spaced ``dy`` apart
    :param dim_names: (optional) names of the two dimensions
    :return: ``xr.DataArray`` of the joint distribution
    """

    # Get the number of bins
    if isinstance(bins, xr.DataArray):
        bins = bins.data

    # Allow passing 'None' arguments in the plot config for certain entries of the range arg
    # This allows clipping only on some dimensions without having to specify every limit
    if ranges is not None:
        ranges = (
            np.array(ranges.data)
            if isinstance(ranges, xr.DataArray)
            else np.array(ranges)
        )
        for idx in range(len(ranges)):
            if None in ranges[idx]:
                ranges[idx] = (
                    [np.min(x), np.max(x)] if idx == 0 else [np.min(y), np.max(y)]
                )
    else:
        ranges = kwargs.pop("range", None)

    # Get the statistics and bin edges
    stat, x_edge, y_edge, _ = scipy.stats.binned_statistic_2d(
        x, y, values, statistic=statistic, bins=bins, range=ranges, **kwargs
    )
    # Normalise the joint distribution, if given
    if normalize:
        if dy is None:
            int_y = [
                scipy.integrate.trapezoid(
                    stat[i][~np.isnan(stat[i])],
                    0.5 * (y_edge[1:] + y_edge[:-1])[~np.isnan(stat[i])],
                )
                for i in range(stat.shape[0])
            ]
        else:
            int_y = [
                scipy.integrate.trapezoid(stat[i][~np.isnan(stat[i])], dx=dy)
                for i in range(stat.shape[0])
            ]

        norm = (
            scipy.integrate.trapezoid(int_y, 0.5 * (x_edge[1:] + x_edge[:-1]))
            if dx is None
            else scipy.integrate.trapezoid(int_y, dx=dx)
        )
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
    joint: xr.DataArray,
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
    marginal = np.array([])
    for p in joint.coords[parameter]:
        _y, _x = joint.sel({parameter: p}).data, joint.coords[integration_coord]
        if scale_y_bins and not np.isnan(_y).all():
            _f = np.nanmax(_y) - np.nanmin(_y)
            _f = 1.0 / _f if _f != 0 else 1.0
        else:
            _f = 1.0
        marginal = np.append(
            marginal,
            _f * scipy.integrate.trapezoid(_y[~np.isnan(_y)], _x[~np.isnan(_y)]),
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
            y=(["bin_idx"], marginal),
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
    bins: Union[int, xr.DataArray] = None,
    ranges: Union[Sequence, xr.DataArray] = None,
    *,
    parameter: str = "x",
    normalize: Union[bool, float] = True,
    scale_y_bins: bool = False,
    **kwargs,
) -> xr.Dataset:
    """
    Computes a marginal directly from a ``xr.DataArray`` of x-values and a ``xr.DataArray`` of probabilities by first
    computing the joint distribution and then marginalising over the probability. This way, points that are sampled
    multiple times only contribute once to the marginal, which is not a representation of the frequency with which
    each point is sampled, but of the calculated likelihood function.

    :param x: array of samples of the first variable (the parameter estimates)
    :param prob: array of samples of (unnormalised) probability values
    :param bins: bins to use for both dimensions
    :param range: range to use for both dimensions. Defaults to the minimum and maximum along each dimension
    :param parameter: the parameter over which to marginalise. Defaults to the first dimension.
    :param normalize: whether to normalize the marginal
    :param scale_y_bins: whether to scale the integration over y by range of the given values (y_max - y_min)
    :param kwargs: other kwargs, passed to ``joint_2D``
    :return: ``xr.Dataset`` of the marginal densities
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
    dim_names: Sequence = None,
    **kwargs,
) -> xr.DataArray:
    """
    Computes the d-dimensional joint distribution of a dataset of parameters by calling ``scipy.stats.binned_statistic_dd``.
    This function can handle at most 32 parameters. A statistic for each bin is returned (mean by default).

    :param sample: ``xr.DataArray`` of samples of shape ``(N, D)``
    :param values: ``xr.DataArray`` of values to be binned, of shape ``(D, )``
    :param bins: bins argument to ``scipy.binned_statistic_dd``
    :param ranges: range argument to ``scipy.binned_statistic_dd``
    :param statistic: (optional) ``statistic`` argument to ``scipy.binned_statistic_2d``
    :param normalize: (not implemented) whether to normalize the joint (False by default),
        and the normalisation value (1 by default)
    :param dim_names: (optional) names of the two dimensions
    :return: ``xr.Dataset`` of the joint distribution
    """
    if normalize:
        raise NotImplementedError(
            "Normalisation for d-dimensional joints is not yet implemented!"
        )

    # Get the number of bins
    if isinstance(bins, xr.DataArray):
        bins = bins.data

    dim_names = (
        sample.coords[list(sample.dims)[-1]].data if dim_names is None else dim_names
    )

    # Allow passing 'None' arguments in the plot config for certain entries of the range arg
    # This allows clipping only on some dimensions without having to specify every limit
    if ranges is not None:
        ranges = ranges.data if isinstance(ranges, xr.DataArray) else ranges
        for idx in range(len(ranges)):
            if None in ranges[idx]:
                ranges[idx] = [np.min(sample.coords[idx]), np.max(sample.coords[idx])]
    else:
        ranges = kwargs.pop("range", None)

    # Get the statistics and bin edges
    stat, bin_edges, _ = scipy.stats.binned_statistic_dd(
        sample, values, statistic=statistic, bins=bins, range=ranges, **kwargs
    )

    return xr.DataArray(
        data=stat,
        dims=dim_names,
        coords={dim_names[i]: 0.5 * (b[1:] + b[:-1]) for i, b in enumerate(bin_edges)},
        name="joint",
    )


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
