from operator import itemgetter
from typing import Sequence

import numpy as np
import scipy.signal
import xarray as xr

from utopya.eval import is_operation

# --- Custom DAG operations for the NeuralABM model --------------------------------------------------------------------


def apply_along_dim(func):
    def _apply_along_axis(
        data: xr.Dataset,
        *args,
        loss: xr.Dataset = None,
        along_dim: str = None,
        labels: list = None,
        **kwargs,
    ):
        """Decorator which allows for applying a function, which acts on an array-like, along a dimension of a
        xarray.Dataset.

        :param data: xr.Dataset containing the parameter estimates and (optionally) loss data
        :param loss: (optional) xr.Dataset containing separate loss data
        :param along_dim: the dimension along with to apply the operation
        :param labels: if passed, relabels the along_dim coordinates
        :param args: additional args, passed to function
        :param kwargs: additional kwargs, passed to function
        :return: xr.Dataset
        """
        if along_dim is not None:
            dsets = []
            for idx, val in enumerate(data[along_dim]):
                dsets.append(
                    func(
                        data.sel({along_dim: val}),
                        loss.sel({along_dim: val}) if loss is not None else None,
                        coords={along_dim: val}
                        if labels is None
                        else {along_dim: labels[idx]},
                        *args,
                        **kwargs,
                    )
                )

            return xr.concat(dsets, dim=along_dim, coords="all")

        else:
            return func(data, loss, *args, **kwargs)

    return _apply_along_axis


@is_operation("NeuralABM.compute_marginals")
@apply_along_dim
def get_marginals(
    data: xr.Dataset,
    *_,
    coords: dict = None,
    bins: int = 100,
    clip: tuple = [-np.inf, +np.inf],
    **__,
) -> xr.Dataset:
    """Sorts the data into bins and calculates marginal densities by summing over each bin entry.

    :param data: the data
    :param coords: (optional) the coordinates to give the new dataset
    :param bins: number of bins
    :param clip: (optional) clip the data to a certain range
    :return: a an xr.Dataset of the marginal densities
    """

    coords = data.coords if coords is None else coords
    coords.update({"bin_idx": np.arange(0, bins, 1)})

    # Collect all points into a list of tuples and sort by their x value
    zipped_pairs = sorted(
        np.array(
            [
                z
                for z in list(
                    zip(data["param1"].values.flatten(), data["loss"].values.flatten())
                )
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
        data_vars=dict(prob=("bin_idx", y), param1=("bin_idx", x)), coords=coords
    )


@is_operation("NeuralABM.compute_joint_density")
@apply_along_dim
def joint_density(
    data: xr.Dataset,
    *,
    coords: dict = None,
    bins: tuple = [100, 100],
    clip: tuple = [[-np.inf, +np.inf], [-np.inf, +np.inf]],
) -> xr.Dataset:
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
            [
                z
                for z in list(
                    zip(
                        data["param1"].values.flatten(),
                        data["param2"].values.flatten(),
                        data["loss"].values.flatten(),
                    )
                )
                if not (
                    np.isnan(z[0])
                    or np.isnan(z[1])
                    or np.isnan(z[2])
                    or z[0] < clip[0][0]
                    or z[0] > clip[0][1]
                    or z[1] < clip[1][0]
                    or z[1] > clip[1][1]
                )
            ]
        ),
        key=itemgetter(0, 1, 2),
    )

    # Create bins
    x = np.linspace(
        min(data["param1"].values.flatten()),
        max(data["param1"].values.flatten()),
        bins[0],
    )
    y = np.linspace(
        min(data["param2"].values.flatten()),
        max(data["param2"].values.flatten()),
        bins[1],
    )
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
    z /= np.sum(z) * dx * dy

    coords.update({"x": x, "y": y})

    # Combine into a xr.Dataset
    return xr.Dataset(data_vars=dict(prob=(["x", "y"], z)), coords=coords)


@is_operation("NeuralABM.compute_mode")
@apply_along_dim
def compute_mode(
    data: xr.Dataset,
    *,
    coords: dict = None,
    x: str = "param1",
    p: str = "prob",
    dim: str = "bin_idx",
):
    """Computes the x-coordinate of the mode of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    idx_max = data[p].idxmax(dim=dim)
    mode = data[x].sel({dim: idx_max})

    return xr.Dataset(data_vars=dict(mode=mode), coords=coords)


@is_operation("NeuralABM.compute_mean")
@apply_along_dim
def compute_mean(
    data: xr.Dataset, *, coords: dict = None, x: str = "param1", p: str = "prob"
):
    """Computes the mean of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    dx = data[x].values[1] - data[x].values[0]
    mean = (data[x].values * data[p].values * dx).sum()

    return xr.Dataset(data_vars=dict(mean=mean), coords=coords)


@is_operation("NeuralABM.compute_std")
@apply_along_dim
def compute_std(
    data: xr.Dataset, *, coords: dict = None, x: str = "param1", p: str = "prob"
):
    """Computes the standard deviation of a one-dimensional dataset consisting of x-values and probabilities"""
    coords = data.coords if coords is None else coords

    dx = data[x].values[1] - data[x].values[0]
    mean = (data[x].values * data[p].values * dx).sum()
    std = np.sqrt((dx * data[p].values * (data[x].values - mean) ** 2).sum())

    return xr.Dataset(data_vars=dict(std=std), coords=coords)


@is_operation("NeuralABM.compute_avg_peak_widths")
@apply_along_dim
def compute_avg_peak_widths(
    data: xr.Dataset, *, coords: dict = None, dim: str = "prob", **kwargs
) -> xr.Dataset:
    """Computes the average peak width using the scipy.signal.peaks function

    :param data: the dataset
    :param coords: (optional) coordinates to use for the returned dataset
    :param dim: (optional) the dimension along which to look for peaks
    :param kwargs: (optional) additional kwargs, passed to scipy.signal.find_peaks
    :return: xr.Dataset of mean peak width and standard deviation
    """

    peaks = scipy.signal.find_peaks(data[dim], **kwargs)
    mean, std = np.mean(peaks[1]["widths"]), np.std(peaks[1]["widths"])

    return xr.Dataset(data_vars=dict(mean=mean, std=std), coords=coords)


@is_operation("NeuralABM.hist")
def hist(
    ds: xr.DataArray, axis: int = 1, *args, bins: Sequence, **kwargs
) -> xr.DataArray:
    def _hist(obj, *args, **kwargs):

        return np.histogram(obj, *args, **kwargs)[0].astype(float)

    data = np.apply_along_axis(_hist, axis, ds, *args, bins, **kwargs)
    dim_0 = list(ds.sizes)[0]
    return xr.DataArray(
        data,
        dims=[dim_0, "bin_center"],
        coords={
            dim_0: ds.coords[list(ds.sizes)[0]],
            "bin_center": bins[:-1] + (np.subtract(bins[1:], bins[:-1])) / 2,
        },
    )


@is_operation("NeuralABM.flatten_dims")
@apply_along_dim
def flatten_dims(ds: xr.DataArray, loss, dim, *args, **kwargs):
    """Flattens dimensions of a dataarray into a new datarray"""

    key, params = list(dim.keys())[0], list(dim.values())[0]
    ds = ds.stack(dim).drop_vars(params)

    return ds.assign_coords({key: np.arange(len(ds.coords[key]))}).transpose(key, ...)


@is_operation("NeuralABM.normalise_degrees_to_edges")
@apply_along_dim
def normalise_degrees_to_edges(ds, *_, **__):

    norms = np.expand_dims(
        ds.sum("bin_center") * np.diff(ds.coords["bin_center"])[0], -1
    )
    # norms = np.expand_dims(np.sum(ds.data * np.expand_dims(ds.coords["bin_center"], 0), axis=1), -1)
    ds.data = ds.data / np.where(norms != 0, norms, 1)
    return ds


@is_operation("NeuralABM.Hellinger_distance")
@apply_along_dim
def Hellinger_distance(P1, loss, P2, x: str = "bin_center", *_, **__):

    return np.square(np.sqrt(P1) - np.sqrt(P2)).sum(x)


@is_operation("NeuralABM.relative_entropy")
@apply_along_dim
def relative_entropy(P1, loss, P2, x: str = "bin_center", *_, **__):

    return np.abs(
        P1 * np.log(xr.where(P1 != 0, P1, 1.0) / xr.where(P2 != 0, P2, 1.0))
    ).sum(x)


@is_operation("NeuralABM.marginal_of_density")
@apply_along_dim
def marginal_of_density(
    vals: xr.DataArray,
    loss: xr.DataArray,
    *,
    coords: dict = {},
    MLE_index: int = None,
    error: str = "standard",
) -> xr.Dataset:

    n_samples, n_bins = list(vals.sizes.values())[:]

    # Calculate the mean of each bin
    means = np.sum(vals.data * loss.data, axis=0)

    # Calculate the uncertainty of each bin
    if error.lower() == "standard":
        err = np.square(vals.data - np.resize(means, (1, n_bins)))
        err = np.sqrt(np.sum(err * loss.data, axis=0))

    elif error.lower() == "hellinger":
        err = np.sqrt(vals.data) - np.sqrt(np.resize(means, (1, n_bins)))
        err = np.sum(np.square(err) * loss.data, axis=0)

    coords.update(dict(bin_idx=np.arange(n_bins)))

    return xr.Dataset(
        data_vars=dict(
            bin_center=("bin_idx", vals.coords["bin_center"].data),
            y=("bin_idx", means),
            yerr=("bin_idx", err),
            MLE=("bin_idx", vals.data[MLE_index]),
        ),
        coords=coords,
    )
