import itertools
from operator import itemgetter

import numpy as np
import scipy.signal
import xarray as xr

from utopya.eval import is_operation

# --- Custom DAG operations for the NeuralABM model --------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# DECORATOR
# ----------------------------------------------------------------------------------------------------------------------


def apply_along_dim(func):
    def _apply_along_axes(
        data: xr.Dataset,
        along_dim: list = None,
        exclude_dim: list = None,
        *args,
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

                # Strip the datasets of all coords
                dsets.append(
                    func(
                        data.sel(
                            dict(
                                [
                                    (excluded_dims[i], excluded_coords[i][idx[i]])
                                    for i in range(len(excluded_dims))
                                ]
                            ),
                            drop=True,
                        ),
                        *args,
                        **kwargs,
                    ).expand_dims(
                        dim=dict(
                            [
                                (excluded_dims[i], [excluded_coords[i][idx[i]]])
                                for i in range(len(excluded_dims))
                            ]
                        )
                    )
                )

            # Merge the datasets into one and return
            return xr.merge(dsets)

        else:
            return func(data, *args, **kwargs)

    return _apply_along_axes


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


# ----------------------------------------------------------------------------------------------------------------------
# MARGINAL DENSITY
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

    # Sort x into bins; cumulatively gather y-values
    for point in zipped_pairs:
        while point[0] > x[bin_no]:
            bin_no += 1
        y[bin_no - 1] += point[1]

    # Normalise y to 1
    y /= np.sum(y) * dx

    # Combine into a xr.Dataset
    return xr.Dataset(
        data_vars=dict(p=("bin_idx", y), x=("bin_idx", x)),
        coords={"bin_idx": np.arange(0, bins, 1)},
    )
