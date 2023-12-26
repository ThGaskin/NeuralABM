import itertools
from typing import Sequence, Union, Tuple

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR CUSTOM DAG OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


def apply_along_dim(func):
    def _apply_along_axes(
        *args,
        along_dim: Sequence = None,
        exclude_dim: Sequence = None,
        **kwargs,
    ):
        """Decorator which allows for applying a function, acting on aligned array-likes, along dimensions of
        xarray objects. The datasets must be aligned. All functions using this header should therefore only take
        xarray objects as arguments that can be indexed along common dimensions. All other arguments should be keywords.

        :param args: Sequence of xarray objects (``xr.Dataset`` or ``xr.DataArray``) which are to be aligned
        :param along_dim: the dimensions along with to apply the operation
        :param exclude_dim: the dimensions to exclude. This is an alternative to providing the 'along_dim' argument.
            Cannot provide both 'along_dim' and 'exclude_dim'
        :param kwargs: passed to function
        :return: if ``along_dim`` or ``exclude_dim`` are given, returns a ``xr.Dataset`` of merged arrays, else returns
            the return type of ``func``.
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


def _hist(
    obj, *, normalize: Union[bool, float] = None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies ``numpy.histogram`` along an axis of an object and returns the counts and bin centres.
    If specified, the counts are normalised. Returns the counts and bin centres (not bin edges!)
    :param obj: data to bin
    :param normalize: (optional) whether to normalise the counts; can be a boolean or a float, in which
        case the float is interpreted as the normalisation constant.
    :param kwargs: passed on to ``np.histogram``
    :return: bin counts and bin centres
    """
    _counts, _edges = np.histogram(obj, **kwargs)
    _counts = _counts.astype(float)
    _bin_centres = 0.5 * (_edges[:-1] + _edges[1:])

    # Normalise, if given
    if normalize:
        norm = np.nansum(_counts)
        norm = 1.0 if norm == 0.0 else norm
        _counts /= norm if isinstance(normalize, bool) else norm / normalize
    return _counts, _bin_centres


def _get_hist_bins_ranges(ds, bins, ranges, axis):
    """Returns histogram bins and ranges in such a way that they can be passed to a histogram function. Bins are
    converted into numpy arrays, and ``None`` entries in the range are converted into the minimum or maximum of
    the data.

    :param ds: dataset to be binned
    :param bins: ``bins`` argument to the histogramming function; if an ``xr.DataArray`` object, is converted
        into a numpy array
    :param ranges: ``ranges`` argument to the histogramming function; if ``None``, is filled with minimum and maximum
        value of the data.
    :param axis: axis along which the histogramming will be applied
    :return: bins and ranges for the histogram
    """

    # Convert the bins from an xarray object to a numpy object, if required
    if isinstance(bins, xr.DataArray):
        bins = bins.data

    # Fill ``None`` entries in the ranges
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


def _interpolate(
    _p: xr.DataArray, _q: xr.DataArray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolates two one-dimensional densities _p and _q onto their common support, with the mesh size given by the sum of the
    individual mesh sizes. ``_p`` and ``_q`` must be one-dimensional."""

    # Get the coordinate dimensions
    _dim1, _dim2 = list(_p.coords.keys())[0], list(_q.coords.keys())[0]

    # Return densities if they are already equal
    if len(_p.coords[_dim1].data) == len(_q.coords[_dim2].data):
        if all(_p.coords[_dim1].data == _q.coords[_dim2].data):
            return _p.data, _q.data, _p.coords[_dim1].data

    # Get the common support
    _x_min, _x_max = np.max(
        [_p.coords[_dim1][0].item(), _q.coords[_dim2][0].item()]
    ), np.min([_p.coords[_dim1][-1].item(), _q.coords[_dim2][-1].item()])

    # Interpolate the functions on the intersection of their support
    _grid = np.linspace(_x_min, _x_max, len(_p.coords[_dim1]) + len(_q.coords[_dim2]))
    _p_interp = np.interp(_grid, _p.coords[_dim1], _p)
    _q_interp = np.interp(_grid, _q.coords[_dim2], _q)

    return _p_interp, _q_interp, _grid
