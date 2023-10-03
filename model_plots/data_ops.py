import itertools
from operator import itemgetter
from typing import Any, Union

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
def concat(objs: list, name: str, dims: list, *args, **kwargs):
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
    ds1: xr.Dataset, ds2: xr.Dataset, *, x: str = "x", p: str = "loss"
) -> xr.Dataset:
    """Broadcasts together two datasets and returns a dataset with the variables of the datasets set as dimensions.
    This is typically used to broadcast together a dataset of parameters of dimension (D, N) and a dataset of associated
    loss values of dimension (N, ) into a new dataset of dimension (D, 2, N), where each parameter now has the loss
    value associated with it, allowing for easy computation of marginals etc.

    :param ds1: the first dataset
    :param ds1: the second dataset
    :param x, p: names for the new dimensions of the variables of the datasets
    :return: an xr.Dataset, with the previous variables now dimensions of the dataset
    """
    return xr.broadcast(xr.Dataset({x: ds1, p: ds2}))[0]


# ----------------------------------------------------------------------------------------------------------------------
# BASIC STATISTICS FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("mean")
@apply_along_dim
def mean(
    data: xr.Dataset,
    *,
    x: str,
    p: str,
) -> xr.Dataset:
    """Computes the mean of a one-dimensional dataset consisting of x values and associated binned probabilities,
    (x, p(x)). Since the probabilities are binned, they are normalised such that sum p(x)dx = 1

    :param data: the dataset
    :param x: the x-value dimension
    :param p: the name of the probability dimension along which to select the mode.
    :return: the mean of the dataset
    """

    # Get the x differential
    dx = abs(data[x].values[1] - data[x].values[0]) if len(data[x].values) > 1 else 1

    # Calculate the mean
    mean = (data[x].values * data[p].values * dx).sum()

    # Return
    return xr.Dataset(data_vars=dict(mean=mean))


@is_operation("std")
@apply_along_dim
def std(
    data: xr.Dataset,
    *,
    x: str,
    p: str,
) -> xr.Dataset:
    """Computes the standard deviation of a one-dimensional dataset consisting of x values and associated binned
    probabilities, (x, p(x)). Since the probabilities are binned, they are normalised such that sum p(x)dx = 1.

    :param data: the dataset
    :param x: the x-value dimension
    :param p: the name of the probability dimension along which to select the mode.
    :return: the standard deviation of the dataset
    """

    # Get the x differential
    dx = abs(data[x].values[1] - data[x].values[0]) if len(data[x].values) > 1 else 1

    # Calculate the mean
    m = mean(data, x=x, p=p).to_array().data
    std = np.sqrt((dx * data[p].values * (data[x].values - m) ** 2).sum())

    # Return
    return xr.Dataset(data_vars=dict(std=std))


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

    # Get the differential
    dx = abs(data[x].values[1] - data[x].values[0]) if len(data[x].values) > 1 else 1

    # Calculate the mean
    mu = mean(data, x=x, p=p).to_array().data

    # Calculate the index of t
    t_index = np.argmin(np.abs(data[x] - t).data)

    # Calculate the p-value depending on the location of t
    if t >= mu:
        return xr.Dataset(data_vars=dict(p_value=data[p][t_index:].sum() * dx))
    else:
        return xr.Dataset(data_vars=dict(p_value=data[p][:t_index].sum() * dx))


@is_operation("hist")
def hist(ds: xr.DataArray, *args, bins, along_dim: list = None, **kwargs) -> xr.Dataset:
    """Applies the numpy.histogram function along one axis of an xr.Dataset. This is significantly faster than
    the 'hist_ndim' function, since it circumvents spliting and re-combining multiple Datasets, but is thus only
    applicable to the one-dimensional case.

    :param ds: the DataArray on which to apply the histogram function
    :param along_dim: the name of the dimension along which to apply hist. Is passed as a list to ensure consistency
        with the syntax of other functions, but can only take a single argument.
    :param bins: the bins to use
    :param args, kwargs: passed to np.histogram
    """
    if along_dim and len(along_dim) > 1:
        raise ValueError(
            "Cannot use the 'hist' function for multidimensional histogram operations!"
            "Use 'hist_ndim' instead."
        )

    def _hist(obj, *args, **kwargs):
        return np.histogram(obj, *args, **kwargs)[0].astype(float)

    along_dim = along_dim[0] if along_dim else ds.dims[0]
    axis = list(ds.dims).index(along_dim)

    data: np.ndarray = np.apply_along_axis(_hist, axis, ds, *args, bins, **kwargs)

    coords = dict(ds.coords)
    coords.update({along_dim: bins[:-1] + (np.subtract(bins[1:], bins[:-1])) / 2})

    # Get the name of the dimension
    res = xr.Dataset(data_vars={ds.name: (list(ds.sizes.keys()), data)}, coords=coords)

    return res.rename({along_dim: "bin_center"})


@is_operation("hist_ndim")
@apply_along_dim
def hist_ndim(
    ds: Union[xr.DataArray, xr.Dataset], bins, axis: int = -1, **kwargs
) -> xr.Dataset:
    """Same as the 'hist' function but using the apply_along_dim decorator to allow histogramming along multiple
    dimensions. Is significantly slower than 'hist' due to the splitting and merging operations.

    :param ds: the DataArray on which to apply the histogram function
    :param axis: the axis along which to apply np.histogram. By default this happens on the innermost axis.
    :param bins: the bins to use
    :param args, kwargs: passed to np.histogram
    """

    # Numpy operations only work on xr.Datasets
    if isinstance(ds, xr.Dataset):
        ds = ds.to_array().squeeze()

    def _hist(obj, **_kwargs):
        # Applies numpy histogram along an axis and returns only the counts
        return np.histogram(obj, **_kwargs)[0].astype(float)

    # Get the name of the dimension
    dim = ds.name if not None else "_variable"

    # Apply the histogram function along the axis
    data = np.apply_along_axis(_hist, axis, ds, bins, **kwargs)

    return xr.Dataset(
        data_vars={dim: ("bin_center", data)},
        coords={"bin_center": np.add(bins[1:], bins[:-1]) / 2},
    )


# ----------------------------------------------------------------------------------------------------------------------
# PROBABILITY DENSITY OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


@is_operation("compute_joint")
@apply_along_dim
def compute_joint(
    data: xr.Dataset,
    p: xr.DataArray,
    *,
    normalize: bool = False,
    differential: float = None,
    statistic: str,
    **kwargs,
) -> xr.Dataset:
    """Computes the joint distribution of a dataset of parameters by calling the scipy.stats.binned_statistic_dd
    function. This function is computationally expensive. It can at most handle 32 dimensional-data,
    and is thus not applicable to high dimensional models (e.g. network learning). In addition, bin counts higher than
    200-300 typically lead to memory overflow, and so fine meshes cannot be achieved. If marginals are
    required it is recommended to use the `compute_marginal` function directly.

    The function returns a statistic for each bin (typically the mean), as well as a standard deviation.

    :param data: dataset of parameter estimates
    :param p: dataset of associated likelihoods
    :param normalize: whether to normalize the joint (False by default)
    :param differential: the spacial differential dx to use for normalisation. Defaults to the grid spacing
    :return: an xr.Dataset of the joint distribution
    """

    # Get the statisics and bin edges
    stat, bin_edges, _ = scipy.stats.binned_statistic_dd(
        data, p, statistic=statistic, **kwargs
    )
    # Also calculate the standard deviation from the mean for each bin, for convenient plotting
    std, _, _ = scipy.stats.binned_statistic_dd(data, p, statistic="std", **kwargs)

    # Normalise the joint distribution, if given
    if normalize:
        differential = (
            np.prod([a[1] - a[0] for a in bin_edges])
            if differential is None
            else differential
        )
        norm = np.nansum(stat) * differential
    else:
        norm = 1

    # Combine into a xr.Dataset
    parameter_coords = data.coords["parameter"].data
    return xr.Dataset(
        data_vars={
            statistic: (parameter_coords, stat / norm),
            "std": (parameter_coords, std / norm),
        },
        coords=dict(
            (parameter_coords[_], 0.5 * (bin_edges[_][1:] + bin_edges[_][:-1]))
            for _ in range(len(parameter_coords))
        ),
    )


@is_operation("marginal_from_joint")
@apply_along_dim
def marginal_from_joint(
    joint: xr.Dataset,
    *,
    parameter: Union[str, list],
    normalize: bool = True,
) -> xr.Dataset:
    """Computes the marginal of a parameter or several parameters from a given joint distribution by summing
    over a joint distribution. If specified, normalises the marginal to 1. This requires first calculating a potentially
    high-dimensional joint distribution, typically using the scripy.stats.binned_statistic_dd function, which is
    computationally expensive and not recommended. Instead, calculate marginals from the parameter estimates directly
    using the "compute_marginal" function

    :param joint: joint distribution of parameter estimates
    :param parameter: parameter over which to marginalise; can be a parameter name or 'all'
    :param normalize: whether to normalise the marginal (True by default)
    :return: an xr.Dataset of the marginal distribution
    """

    def _marginal_1d(_joint: xr.Dataset, _parameter: str) -> xr.Dataset:
        """Computes the marginal of a single parameter"""
        _parameters = _joint.coords
        for _p in list(_parameters):
            if _p != _parameter:
                _joint = _joint.sum(_p)
        if normalize:
            _dx = (
                _joint["mean"].coords[_parameter].data[1]
                - _joint["mean"].coords[_parameter].data[0]
            )
            _joint /= _joint["mean"].sum(skipna=True) * _dx
        _x_vals = joint.coords[_parameter].values
        _joint = (
            _joint.assign_coords({_parameter: np.arange(len(_x_vals))})
            .rename({_parameter: "bin_idx"})
            .assign(x=("bin_idx", _x_vals))
        )
        return _joint

    if isinstance(parameter, str):
        if parameter == "all":
            parameter = list(joint.coords)
        else:
            parameter = [parameter]

    # Multiple marginals can only be merged into a single Dataset if all parameter lengths are equal
    if not all(
        [len(joint.coords[p]) == len(joint.coords[parameter[0]]) for p in parameter]
    ):
        raise ValueError(
            "The different parameters have different bin sizes and cannot be merged into a single "
            "xr.Dataset!"
        )

    return xr.concat(
        [_marginal_1d(joint, p).expand_dims({"parameter": [p]}) for p in parameter],
        dim="parameter",
    )


@is_operation("compute_marginal")
@apply_along_dim
def compute_marginal(
    data: xr.Dataset,
    bins: Any = 100,
    *,
    x: str,
    p: str,
    statistic: Union[str, callable] = "mean",
    normalize: bool = True,
    aggregate: int = None,
) -> xr.Dataset:
    """Calculates the marginal density of a single parameter. This function avoids having to first calculate a high-
    dimensional joint density function, which can become computationally prohibitively expensive. Instead, in order to
    estimate the probability exp(-J) of each bin, the parameter is first binned into a very fine grid, ensuring that
    any two points in the same bin are very likely to be the same point. The probability of each bin is then estimated
    using a statistic function -- typically the mean -- and the grid can then be aggregated into a coarser grid by
    summing over adjacenct bins (using the 'aggregate' keyword) or by smoothing over the fine grid.
    (done by the plot function).

    :param data: the dataset, containing parameter estimates
    :param x: the name of the parameter variable
    :param p: the name of the probability variable
    :param bins: number and range of bins of the output. Can either be a single int, in which case it is interpreted as
        the number of bins; can be a 2-tuple of ints, in which case it is interpreted as a range; or it can be a
        three-tuple, in which case it is interpreted as a range and the number of bins.
    :param statistic: how to process identical points in the same mean
    :param normalize: whether to normalise the marginal to 1

    :return: an xr.Dataset of the marginal densities
    """

    # Dictionary of available statistics functions
    _STAT_FUNCS: dict = {
        "sum": np.nansum,
        "mean": np.nanmean,
        "max": np.max,
        "min": np.min,
        "count": len,
    }

    # Get the range of the bins and number of bins (100 by default)
    if isinstance(bins, int):
        bins = [bins]
    if isinstance(bins, (xr.Dataset, xr.DataArray)):
        bins = bins.to_numpy()

    if len(bins) == 1:
        bins = bins[0]
        clip = [-np.inf, +np.inf]
    elif len(bins) == 2:
        clip = bins
        bins = int(100)
    else:
        clip, bins = bins[0:2], int(bins[-1])

    # Collect all points into a list of tuples and sort by their x value
    zipped_pairs = sorted(
        np.array(
            [
                z
                for z in list(zip(data[x].values.flatten(), data[p].values.flatten()))
                if not (
                    np.isnan(z[0]) or np.isnan(z[1]) or z[0] < clip[0] or z[0] > clip[1]
                )
            ]
        ),
        key=itemgetter(0),
    )

    # Create bins
    x, y = np.linspace(zipped_pairs[0][0], zipped_pairs[-1][0], bins), dict(
        (_, []) for _ in range(bins)
    )
    bin_no = 1

    for point in zipped_pairs:
        while point[0] > x[bin_no]:
            bin_no += 1
        y[bin_no - 1].append(point[1])

    # Aggregate points in the same bin using the given statistic (mean by default)
    _stat_func = _STAT_FUNCS[statistic] if isinstance(statistic, str) else statistic
    y = np.array([_stat_func(_) if _ else 0 for _ in y.values()])

    # If given, coarsen the bins
    if aggregate:
        x = x[int(aggregate / 2) :: aggregate]
        y = [
            np.sum(y[i : i + aggregate])
            for i in range(0, len(y) - aggregate + 1, aggregate)
        ]

    # Calculate the differential and normalise to 1
    if normalize:
        dx = x[1] - x[0]
        y /= np.nansum(y) * dx

    # Combine into a xr.Dataset
    return xr.Dataset(
        data_vars=dict(p=("bin_idx", y), x=("bin_idx", x)),
        coords={"bin_idx": np.arange(0, len(x), 1)},
    )


@is_operation("Hellinger_distance")
@apply_along_dim
def Hellinger_distance(
    p: Union[xr.Dataset, xr.DataArray],
    q: Union[xr.Dataset, xr.DataArray],
    *,
    sum: bool = True,
    x: str = None,
) -> xr.Dataset:
    """Calculates the pointwise Hellinger distance between two distributions p and q, defined as

        d_H(p, q)(x) = sqrt(p(x)) - sqrt(q(x))**2

    If p is a dataset, the Hellinger distance is computed for each distribution in the family. If given, the
    total Hellinger distance along a dimension is also calculated by summing along x. If x is not given, the total
    distance is returned.

    :param p: array or dataset of density values, possibly family-wise
    :param q: one-dimensional dataset or array of density values
    :param sum: (optional) whether to calculate the total Hellinger distance
    :param x: (optional) the dimension along which to calculate the total hellinger distance. If not given, sums over all
        dimensions
    :return:
    """
    res = np.square(np.sqrt(p) - np.sqrt(q))
    if sum:
        if x:
            return res.sum(x)
        else:
            # If summing over all dimensions, return as xr.Dataset to allow later stacking
            return xr.Dataset(data_vars=dict(Hellinger_distance=res.sum()))
    else:
        return res


@is_operation("relative_entropy")
@apply_along_dim
def relative_entropy(
    p: Union[xr.Dataset, xr.DataArray],
    q: Union[xr.Dataset, xr.DataArray],
    *,
    sum: bool = True,
    x: str = None,
) -> xr.Dataset:
    """Calculates the relative_entropy distance between two distributions p and q, defined as

        d_H(p, q) = int p(x) log(p(x)/q(x))dx

    Canonically, log(0/0) = log(1/1) = 0. If p is a dataset, the relative entropy distance is computed for each
    distribution  in the family. If given, the total relative entropy along a dimension is also calculated by summing
    along x. If x is not given, the total distance is returned.

    :param p: one-dimensional array of density values
    :param q: one-dimensional array of density values
    :param sum: (optional) whether to calculate the total Hellinger distance
    :param x: (optional) the dimension along which to calculate the total hellinger distance. If not given, sums over all
        dimensions
    :return: the relative entropy between p and q
    """
    res = np.abs(p * np.log(xr.where(p != 0, p, 1.0) / xr.where(q != 0, q, 1.0)))
    if sum:
        if x:
            return res.sum(x)
        else:
            # If summing over all dimensions, return as xr.Dataset to allow later stacking
            return xr.Dataset(data_vars=dict(relative_entropy=res.sum()))
    else:
        return res


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


@is_operation("marginal_of_density")
@apply_along_dim
def marginal_of_density(
    densities: Union[xr.DataArray, xr.Dataset],
    p: xr.DataArray,
    *,
    error: str,
    sample_dim: str = "sample",
) -> Union[xr.Dataset, xr.DataArray]:
    """Calculates the marginal density with an error over a family of distributions, each with an associated probability.
    The error can be calculated either as an L2 distance, a Hellinger distance, or a relative entropy.
    The uncertainty is the expectation value of the chosen metric for each value, that is

        d(x) = int d(x, a) p(a) da,

    where a is the index of the distribution.

    :param densities: the family of distributions
    :param p: the normalised probability associated with each density
    :param error: the type of error to use. Can be either 'L2', 'Hellinger', or 'relative_entropy'
    :param sample_dim: the name of the dimension indexing the distributions
    :return: an xr.Dataset containing the mean density and error
    """
    # Calculate the mean of each x value
    means = (densities * p).sum(sample_dim)

    # Calculate the uncertainty of each bin
    if error.lower() == "l2":
        err = (np.sqrt(np.square(densities - means)) * p).sum(sample_dim)

    elif error.lower() == "hellinger":
        err = (np.square(np.sqrt(densities) - np.sqrt(means)) * p).sum(sample_dim)

    elif error.lower() == "relative_entropy":
        err = (
            densities
            * np.log(
                xr.where(densities != 0, p, 1.0) / xr.where(means != 0, means, 1.0)
            )
            * p
        ).sum(sample_dim)

    else:
        raise ValueError(f"Unrecognised error type {error}!")

    # Get the name of the dimension and rename
    if isinstance(means, xr.Dataset):
        dim = list(means.keys())[0]
        return xr.merge([means.rename({dim: "mean"}), err.rename({dim: "err"})])
    else:
        return xr.merge([means.rename("mean"), err.rename("err")])


# ----------------------------------------------------------------------------------------------------------------------
# ADJACENCY MATRIX OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


@is_operation("normalise_to_nw_size")
@apply_along_dim
def normalise_to_nw_size(ds: xr.Dataset, *, x: str) -> xr.Dataset:
    """Normalises a one-dimensional xarray object of node degrees to the total number of edges in the graph, i.e.

        k = k / int k dx

    This is required to compare degree distributions between networks.

    :param degrees: the dataset containing the degrees
    :param x: name of the coordinate dimension
    :return: xarray object of normalised degrees
    """
    norms = ds.sum(x) * np.mean(np.diff(ds.coords[x]))

    ds = ds / xr.where(norms != 0, norms, 1)

    return ds


# ----------------------------------------------------------------------------------------------------------------------
# ADJACENCY MATRIX OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("triangles")
def triangles(
    ds: xr.DataArray,
    offset=0,
    axis1=1,
    axis2=2,
    *args,
    input_core_dims: list = ["j"],
    **kwargs,
):
    """Calculates the number of triangles on each node from an adjacency matrix along one dimension.
    The number of triangles are given by

        t(i) = 1/2 sum a_{ij}_a{jk}a_{ki}

    in the undirected case, which is simply the i-th entry of the diagonal of A**3. This does not use the apply_along_dim
    function is thus fast, but cannot be applied along multiple dimensions.

    :param a: the adjacency matrix
    :param offset, axis1, axis2: passed to numpy.diagonal
    :param input_core_dims: passed to xr.apply_ufunc
    :param args, kwargs: additional args and kwargs passed to np.linalg.matrix_power

    """

    res = xr.apply_ufunc(np.linalg.matrix_power, ds, 3, *args, **kwargs)
    return xr.apply_ufunc(
        np.diagonal,
        res,
        offset,
        axis1,
        axis2,
        input_core_dims=[input_core_dims, [], [], []],
    )


@is_operation("triangles_ndim")
@apply_along_dim
def triangles_ndim(
    a: Union[xr.Dataset, xr.DataArray],
    offset=0,
    axis1=0,
    axis2=1,
    *args,
    input_core_dims: list = ["j"],
    **kwargs,
) -> Union[xr.DataArray, xr.Dataset]:
    """Calculates the number of triangles on each node from an adjacency matrix. The number of triangles
    are given by

        t(i) = 1/2 sum a_{ij}_a{jk}a_{ki}

    in the undirected case, which is simply the i-th entry of the diagonal of A**3. This uses the apply_along_dim
    function and can thus be applied along any dimension, but is slower than 'triangles'.

    :param a: the adjacency matrix
    :param args, kwargs: additional args and kwargs passed to np.linalg.matrix_power

    """
    res = xr.apply_ufunc(
        np.diagonal,
        xr.apply_ufunc(np.linalg.matrix_power, a, 3, *args, **kwargs),
        offset,
        axis1,
        axis2,
        input_core_dims=[input_core_dims, [], [], []],
    )

    if isinstance(res, xr.DataArray):
        return res.rename("triangles")
    else:
        return res
