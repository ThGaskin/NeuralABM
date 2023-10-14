import itertools
from operator import itemgetter
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

    return xr.Dataset(
        data_vars=dict(mean=scipy.integrate.trapezoid(data[p] * data[x], data[x]))
    )


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

    # Calculate the mean
    m = mean(data, x=x, p=p).to_array().data
    std = np.sqrt(scipy.integrate.trapezoid(data[p] * (data[x] - m) ** 2, data[x]))

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
    ds: Union[xr.DataArray, xr.Dataset],
    bins: Any = 100,
    ranges: Any = None,
    *,
    axis: int = -1,
    normalize: Union[int, bool] = None,
    **kwargs,
) -> xr.Dataset:
    """Same as the 'hist' function but using the apply_along_dim decorator to allow histogramming along multiple
    dimensions. Is significantly slower than 'hist' due to the splitting and merging operations.

    :param ds: the DataArray on which to apply the histogram function
    :param axis: the axis along which to apply np.histogram. By default this happens on the innermost axis.
    :param bins: the bins to use
    :param ranges: the range of the bins to use
    :param args, kwargs: passed to np.histogram
    """

    # Get the bins and range objects
    if isinstance(ds, xr.Dataset):
        ds = ds.to_array().squeeze()
    if isinstance(bins, xr.DataArray):
        bins = bins.data
    if ranges is not None:
        ranges = np.array(ranges.data)
        for idx in range(len(ranges)):
            if ranges[idx] is None:
                ranges[idx] = (
                    np.min(ds.data[axis]) if idx == 0 else np.max(ds.data[axis])
                )

    def _hist(obj, **_kwargs):
        # Applies numpy histogram along an axis and returns the counts and bin centres
        _counts, _edges = np.histogram(obj, **_kwargs)
        return _counts.astype(float), np.add(_edges[1:], _edges[:-1]) / 2

    # Get the name of the dimension
    dim = ds.name if ds.name is not None else "_variable"

    # Apply the histogram function along the axis
    counts, bin_centres = np.apply_along_axis(
        _hist, axis, ds, bins=bins, range=ranges, **kwargs
    )

    # Normalize the counts, f given
    if normalize:
        norm = scipy.integrate.trapezoid(counts, bin_centres)
        counts /= norm if isinstance(normalize, bool) else normalize / norm

    return xr.Dataset(
        data_vars={dim: ("bin_idx", counts), "x": ("bin_idx", bin_centres)},
        coords={"bin_idx": np.arange(len(bin_centres))},
    )


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
) -> xr.Dataset:
    """
    Computes a marginal from a two-dimensional joint distribution by summing over one parameter. Normalizes
    the marginal, if specified. NaN values in the joint are skipped when normalising: they are not zero, just unknown.
    Since x-values may differ for different parameters, the x-values are variables in a dataset, not coordinates.
    The coordinates are given by the bin index, thereby allowing marginals across multiple parameters to be combined
    into a single xr.Dataset.
    """

    # Get the integration coordinate
    integration_coord = list(joint.coords)
    integration_coord.remove(parameter)
    integration_coord = integration_coord[0]

    marginal = []
    for i in range(len(joint.coords[parameter])):
        _y, _x = joint.isel({parameter: i}).data, joint.coords[integration_coord]
        marginal.append(scipy.integrate.trapezoid(_y[~np.isnan(_y)], _x[~np.isnan(_y)]))

    # Normalise, if given
    if normalize:
        norm = scipy.integrate.trapezoid(marginal, joint.coords[parameter])
        marginal /= norm if isinstance(normalize, bool) else norm / normalize

    # Return a dataset with x- and y-values as variables, and coordinates given by the bin index
    return xr.Dataset(
        data_vars=dict(
            x=(["bin_idx"], joint.coords[parameter].data),
            marginal=(["bin_idx"], marginal),
        ),
        coords=dict(bin_idx=(["bin_idx"], np.arange(len(joint.data)))),
    )


@is_operation("marginal")
@apply_along_dim
def marginal(
    x: xr.DataArray,
    prob: xr.DataArray,
    bins: xr.DataArray,
    ranges: xr.DataArray,
    *,
    parameter: str = "x",
    normalize: Union[bool, float] = True,
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
    :param kwargs: other kwargs, passed to the joint_2D function
    :return: an xr.Dataset of the marginal densities
    """
    joint = joint_2D(x, prob, prob, bins, ranges, normalize=normalize, **kwargs)
    return marginal_from_joint(joint, parameter=parameter, normalize=normalize)


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

    :param sample: DataArray of values in the first dimension
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


# ----------------------------------------------------------------------------------------------------------------------
# MATRIX SELECTION POWERGRID OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("sel_matrix_indices")
@apply_along_dim
def matrix_indices_sel(
    ds: xr.DataArray, indices: xr.Dataset, drop: bool = False
) -> xr.DataArray:
    """Returns the predictions on the weights of the entries given by indices"""

    ds = ds.isel(i=(indices["i"]), j=(indices["j"]))
    return ds.drop_vars(["i", "j"]) if drop else ds


@is_operation("largest_entry_indices")
@apply_along_dim
def largest_entry_indices(
    ds: xr.DataArray, n: int, *, symmetric: bool = True
) -> xr.Dataset:
    """Returns the 2d-indices of the n largest entries in an adjacency matrix, as well as the corresponding values.
    If the matrix is symmetric, only the upper triangle is considered. Sorted from highest to lowest."""

    if symmetric:
        indices_i, indices_j = np.unravel_index(
            np.argsort(np.triu(ds.data).ravel()), np.shape(ds)
        )
    else:
        indices_i, indices_j = np.unravel_index(
            np.argsort(ds.data.ravel()), np.shape(ds)
        )

    i, j = indices_i[-n:][::-1], indices_j[-n:][::-1]
    vals = ds.data[i, j]

    return xr.Dataset(
        data_vars=dict(i=("idx", i), j=("idx", j), relative_error=("idx", vals)),
        coords=dict(idx=("idx", np.arange(len(i)))),
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
