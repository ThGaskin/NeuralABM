import itertools
from operator import itemgetter
from typing import Union

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr

from utopya.eval import is_operation

# --- Custom DAG operations for the NeuralABM model --------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# DECORATORS
# ----------------------------------------------------------------------------------------------------------------------


def apply_along_dim(func):
    def _apply_along_axes(
        data: xr.Dataset,
        *args,
        along_dim: list = None,
        exclude_dim: list = None,
        **kwargs,
    ):
        """Decorator which allows for applying a function, which acts on an array-like, along dimensions of a
        xarray.Dataset.

        :param data: xr.Dataset
        :param along_dim: the dimensions along with to apply the operation
        :param exclude_dim: the dimensions to exclude. This is an alternative to providing the 'along_dim' argument.
            Cannot provide both 'along_dim' and 'exclude_dim'
        :param args: additional args, passed to function
        :param kwargs: additional kwargs, passed to function
        :return: xr.Dataset
        """
        if along_dim and exclude_dim:
            raise ValueError("Cannot provide both 'along_dim' and 'exclude_dim'!")

        if along_dim is not None or exclude_dim is not None:

            # Get the coordinates for all the dimensions that are to be excluded
            if exclude_dim is None:
                excluded_dims = []
                for c in list(data.coords.keys()):
                    if c not in along_dim:
                        excluded_dims.append(c)
            else:
                excluded_dims = exclude_dim
            excluded_coords = [data.coords[_].data for _ in excluded_dims]

            # Collect the dsets into one dataset
            dsets = []

            # Iterate over all coordinates in the dimensions and apply the function separately
            for idx in itertools.product(*(range(len(_)) for _ in excluded_coords)):

                # Strip the dataset of all coords except the ones along which the function is being
                # applied. Add the coordinates back afterwards and re-merge.
                dsets.append(
                    func(
                        data.sel(
                            {
                                excluded_dims[i]: excluded_coords[i][idx[i]]
                                for i in range(len(excluded_dims))
                            },
                            drop=True,
                        ),
                        *args,
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
            return func(data, *args, **kwargs)

    return _apply_along_axes


def apply_along_dim_2d(func):
    def _apply_along_axes(
        ds1: xr.Dataset,
        ds2: xr.Dataset,
        *args,
        along_dim: list = None,
        exclude_dim: list = None,
        **kwargs,
    ):
        """Decorator which allows for applying a function, which acts on two aligned array-likes, along dimensions of
        the xarray.Datasets. The datasets must be aligned.

        :param ds1: xr.Dataset.
        :param ds2: xr.Dataset. Must be aligned with ds1.
        :param along_dim: the dimensions along with to apply the operation
        :param exclude_dim: the dimensions to exclude. This is an alternative to providing the 'along_dim' argument.
            Cannot provide both 'along_dim' and 'exclude_dim'
        :param args: additional args, passed to function
        :param kwargs: additional kwargs, passed to function
        :return: xr.Dataset
        """
        if along_dim and exclude_dim:
            raise ValueError("Cannot provide both 'along_dim' and 'exclude_dim'!")

        if along_dim is not None or exclude_dim is not None:

            # Get the coordinates for all the dimensions that are to be excluded
            if exclude_dim is None:
                excluded_dims = []
                for c in list(ds1.coords.keys()):
                    if c not in along_dim:
                        excluded_dims.append(c)
            else:
                excluded_dims = exclude_dim
            excluded_coords = [ds1.coords[_].data for _ in excluded_dims]

            # Collect the dsets into one dataset
            dsets = []

            # Iterate over all coordinates in the dimensions and apply the function separately
            for idx in itertools.product(*(range(len(_)) for _ in excluded_coords)):

                # Strip both datasets of all coords except the ones along which the function is being
                # applied. Add the coordinates back afterwards and re-merge.
                dsets.append(
                    func(
                        ds1.sel(
                            {
                                excluded_dims[i]: excluded_coords[i][idx[i]]
                                for i in range(len(excluded_dims))
                            },
                            drop=True,
                        ),
                        ds2.sel(
                            {
                                excluded_dims[j]: excluded_coords[j][idx[j]]
                                for j in range(len(excluded_dims))
                            },
                            drop=True,
                        ),
                        *args,
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
            return func(ds1, ds2, *args, **kwargs)

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
    dims: list,
    new_dim: str,
    *,
    new_coords: list = None,
) -> xr.Dataset:
    """Flattens dimensions of an xr.Dataset into a new dimension. New coordinates can be assigned,
    else the dimension is simply given trivial dimensions. The operation is a combination of stacking and
    subsequently dropping the multiindex.

    :param ds: the xr.Dataset to reshape
    :param dims: list of dims to stack
    :param new_dim: name of the new dimension
    :param new_coords: the coordinates for the new dimension (optional)
    """

    # Stack and drop the dimensions
    ds = ds.stack({new_dim: dims})
    q = set(dims)
    q.add(new_dim)
    ds = ds.drop_vars(q)

    # Add coordinates to new dimension and return
    if new_coords is None:
        return ds.assign_coords({new_dim: np.arange(len(ds.coords[new_dim]))})
    else:
        return ds.assign_coords({new_dim: new_coords})


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
def mode(data: xr.Dataset, *, p: str, **kwargs) -> xr.Dataset:
    """Returns the x value and associated probability of the mode of a one-dimensional dataset
    consisting of x-values and associated probabilities, (x, p(x)).

    :param data: the one-dimensional dataset
    :param p: the name of the probability dimension along which to select the mode.
    :param kwargs: keyword arguments passed to xr.DataArray.idxmax
    :return: the mode of the dataset
    """

    # Get the name of the dimension
    coord = list(data.dims.keys())[0]

    # Get the value of the mode and select it
    x_max = data[p].idxmax(**kwargs).item()
    return data.sel({coord: x_max}).drop_vars(coord)


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
    ds: Union[xr.DataArray, xr.Dataset], *args, bins, axis: int = -1, **kwargs
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

    def _hist(obj, *args, **kwargs):
        # Applies numpy histogram along an axis and returns only the counts
        return np.histogram(obj, *args, **kwargs)[0].astype(float)

    # Get the name of the dimension
    dim = ds.name if not None else "_variable"

    # Apply the histogram function along the axis
    data = np.apply_along_axis(_hist, axis, ds, *args, bins, **kwargs)

    return xr.Dataset(
        data_vars={dim: ("bin_center", data)},
        coords={"bin_center": np.add(bins[1:], bins[:-1]) / 2},
    )


# ----------------------------------------------------------------------------------------------------------------------
# DENSITY OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


@is_operation("compute_marginals")
@apply_along_dim
def marginals(
    data: xr.Dataset,
    *,
    x: str,
    p: str,
    bins: int = 100,
    clip: tuple = [-np.inf, +np.inf],
) -> xr.Dataset:
    """Sorts a dataset containing pairs of estimated parameters and associated probabilities (a, p(a)) into
    bins and calculates the marginal density by summing over each bin entry. All dimensions in the dataset are
    flattened and a one-dimensional dataset returned (with the bin index as the coordinate dimension); for example,
    if the dataset contains estimates from many different seeds, these are all flattened into a single long array
    before marginalising.

    :param data: the dataset, containing parameter estimates
    :param x: the name of the parameter variable
    :param p: the name of the probability variable
    :param bins: number of bins
    :param clip: clip the data to a certain range
    :return: an xr.Dataset of the marginal densities
    """

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
    x, y = np.linspace(zipped_pairs[0][0], zipped_pairs[-1][0], bins), np.zeros(bins)
    dx = x[1] - x[0]
    bin_no = 1

    # # Sort x into bins; cumulatively gather y-values
    # for point in zipped_pairs:
    #     while point[0] > x[bin_no]:
    #         bin_no += 1
    #     y[bin_no - 1] += point[1]

    # Normalise y to 1
    y /= np.sum(y) * dx

    # Combine into a xr.Dataset
    return xr.Dataset(
        data_vars=dict(p=("bin_idx", y), x=("bin_idx", x)),
        coords={"bin_idx": np.arange(0, bins, 1)},
    )


@is_operation("Hellinger_distance")
@apply_along_dim_2d
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
@apply_along_dim_2d
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
@apply_along_dim_2d
def L2_distance(p: xr.DataArray, q: xr.DataArray) -> xr.Dataset:
    """Calculates the L2 distance between two distributions p and q, defined as

        d_H(p, q) = sqrt(int (p(x) - q(x))**2 dx)

    :param p: one-dimensional array of density values
    :param q: one-dimensional array of density values
    :return: the L2 distance between p and q
    """

    coords = data.coords if coords is None else coords

    # Collect all points into a list of tuples and sort first by their x value, then by their y value,
    # filtering out nans and clipping to the specified interval, if passed.
    zipped_pairs = sorted(
        np.array(
            [z for z in list(zip(data['param1'].values.flatten(),
                                 data['param2'].values.flatten(),
                                 data['loss'].values.flatten()))
             if not (np.isnan(z[0]) or np.isnan(z[1]) or np.isnan(z[2])
                     or z[0] < clip[0][0] or z[0] > clip[0][1] or z[1] < clip[1][0] or z[1] > clip[1][1])
             ]
        ), key=itemgetter(0, 1, 2))

    # Create bins
    x = np.linspace(min(data['param1'].values.flatten()), max(data['param1'].values.flatten()), bins[0])
    y = np.linspace(min(data['param2'].values.flatten()), max(data['param2'].values.flatten()), bins[1])
    z = np.zeros(bins)
    dx, dy = (x[1] - x[0]), (y[1] - y[0])
    bin_no_x = 1

    for point in zipped_pairs:

        bin_no_y = 1

        # Find bin in x:
        while point[0] > x[bin_no_x]:
            bin_no_x += 1

        # Find bin in y:
        while point[1] > y[bin_no_y]:
            bin_no_y += 1

        z[bin_no_x - 1][bin_no_y - 1] += point[2]

    # Normalise z to 1
    z /= (np.sum(z) * dx * dy)


    # Combine into a xr.Dataset
    return xr.Dataset(data_vars=dict(prob=(['x', 'y'], z)), coords=coords)


@is_operation('NeuralABM.compute_mode')
@apply_along_dim
def compute_mode(data: xr.Dataset, *, coords: dict = None, x: str = 'param1', p: str = 'prob', dim: str = 'bin_idx'):
    """ Computes the x-coordinate of the mode of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    idx_max = data[p].idxmax(dim=dim)
    mode = data[x].sel({dim: idx_max})

    return xr.Dataset(data_vars=dict(mode=mode), coords=coords)

    return ds

@is_operation('NeuralABM.compute_mean')
@apply_along_dim
def compute_mean(data: xr.Dataset, *, coords: dict = None, x: str = 'param1', p: str = 'prob'):
    """ Computes the mean of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    ds = ds / xr.where(norms != 0, norms, 1)

    return ds

@is_operation('NeuralABM.compute_empirical_mean')
@apply_along_dim
def compute_empirical_mean(data: xr.Dataset, *, coords: dict = None, dims: str = 'seed'):
    """ Computes the mean of a one-dimensional dataset consisting of x-values"""
    # Get time series mean
    mean = data.mean(dim=dims,skipna=True)
    return xr.Dataset(data_vars=dict(mean=mean), coords=mean.coords)


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
