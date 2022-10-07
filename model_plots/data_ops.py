import sys
import numpy as np
from operator import itemgetter
import scipy.signal
import xarray as xr

from utopya.eval import is_operation


# --- Custom DAG operations for the NeuralABM model --------------------------------------------------------------------

def apply_along_dim(func):
    def _apply_along_axis(data: xr.Dataset, along_dim: str = None, labels: list = None, *args, **kwargs):
        """Decorator which allows for applying a function, which acts on an array-like, along a dimension of a
        xarray.Dataset.

        :param data: xr.Dataset
        :param along_dim: the dimension along with to apply the operation
        :param labels: if passed, relabels the along_dim coordinates
        :param args: additional args, passed to function
        :param kwargs: additional kwargs, passed to function
        :return: xr.Dataset
        """
        if along_dim is not None:
            dsets = []
            for idx, val in enumerate(data[along_dim]):
                dsets.append(func(data.sel({along_dim: val}),
                                  coords={along_dim: val} if labels is None else {along_dim: labels[idx]},
                                  *args, **kwargs))

            return xr.concat(dsets, dim=along_dim, coords='all')

        else:
            return func(data, *args, **kwargs)

    return _apply_along_axis


@is_operation('NeuralABM.compute_marginals')
@apply_along_dim
def get_marginals(data: xr.Dataset, *, coords: dict = None, bins: int = 100,
                  clip: tuple = [-np.inf, +np.inf]) -> xr.Dataset:
    """ Sorts the data into bins and calculates marginal densities by summing over each bin entry.

    :param data: the data
    :param coords: (optional) the coordinates to give the new dataset
    :param bins: number of bins
    :param clip: (optional) clip the data to a certain range
    :return: a an xr.Dataset of the marginal densities
    """

    coords = data.coords if coords is None else coords
    coords.update({'bin_idx': np.arange(0, bins, 1)})

    # Collect all points into a list of tuples and sort by their x value
    zipped_pairs = sorted(
        np.array(
            [z for z in list(zip(data['param1'].values.flatten(), data['loss'].values.flatten()))
             if not (np.isnan(z[0]) or np.isnan(z[1]) or z[0] < clip[0] or z[0] > clip[1])]
        ), key=itemgetter(0))

    # Create bins
    x, y = np.linspace(zipped_pairs[0][0], zipped_pairs[-1][0], bins), np.zeros(bins)
    dx = (x[1] - x[0])
    bin_no = 1

    # Sort x into bins; cumulatively gather y-values
    for point in zipped_pairs:
        while point[0] > x[bin_no]:
            bin_no += 1
        y[bin_no - 1] += point[1]

    # Normalise y to 1
    y /= (np.sum(y) * dx)

    # Combine into a xr.Dataset
    return xr.Dataset(data_vars=dict(prob=('bin_idx', y), param1=('bin_idx', x)),
                      coords=coords)


@is_operation('NeuralABM.compute_joint_density')
@apply_along_dim
def joint_density(data: xr.Dataset,
                  *,
                  coords: dict = None,
                  bins: tuple = [100, 100],
                  clip: tuple = [[-np.inf, +np.inf], [-np.inf, +np.inf]]) -> xr.Dataset:
    """Computes the 2d joint probability density function

    :param data: the data
    :param coords: (optional) the coordinates to give the new dataset
    :param bins: number of bins
    :param clip: (optional) clip the data to a certain range
    :return: a an xr.Dataset of the marginal densities
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

    coords.update({'x': x, 'y': y})

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


@is_operation('NeuralABM.compute_mean')
@apply_along_dim
def compute_mean(data: xr.Dataset, *, coords: dict = None, x: str = 'param1', p: str = 'prob'):
    """ Computes the mean of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    dx = data[x].values[1] - data[x].values[0]
    mean = (data[x].values * data[p].values * dx).sum()

    return xr.Dataset(data_vars=dict(mean=mean), coords=coords)

@is_operation('NeuralABM.compute_empirical_mean')
@apply_along_dim
def compute_empirical_mean(data: xr.Dataset, *, coords: dict = None, dims: str = 'seed'):
    """ Computes the mean of a one-dimensional dataset consisting of x-values"""
    # Get time series mean
    mean = data.mean(dim=dims,skipna=True)
    return xr.Dataset(data_vars=dict(mean=mean), coords=mean.coords)


@is_operation('NeuralABM.compute_std')
@apply_along_dim
def compute_std(data: xr.Dataset, *, coords: dict = None, x: str = 'param1', p: str = 'prob'):
    """ Computes the standard deviation of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    dx = data[x].values[1] - data[x].values[0]
    mean = (data[x].values * data[p].values * dx).sum()
    std = np.sqrt((dx * data[p].values * (data[x].values - mean) ** 2).sum())

    return xr.Dataset(data_vars=dict(std=std), coords=coords)


@is_operation('NeuralABM.compute_avg_peak_widths')
@apply_along_dim
def compute_avg_peak_widths(data: xr.Dataset, *, coords: dict = None, dim: str = 'prob', **kwargs) -> xr.Dataset:
    """Computes the average peak width using the scipy.signal.peaks function

    :param data: the dataset
    :param coords: (optional) coordinates to use for the returned dataset
    :param dim: (optional) the dimension along which to look for peaks
    :param kwargs: (optional) additional kwargs, passed to scipy.signal.find_peaks
    :return: xr.Dataset of mean peak width and standard deviation
    """

    peaks = scipy.signal.find_peaks(data[dim], **kwargs)
    mean, std = np.mean(peaks[1]['widths']), np.std(peaks[1]['widths'])

    return xr.Dataset(data_vars=dict(mean=mean, std=std), coords=coords)

