import numpy as np
import xarray as xr

from utopya.eval import is_operation
from utopya.plotting import register_operation

register_operation(name="np.maximum", func=np.maximum)
register_operation(name="np.subtract", func=np.subtract)
register_operation(name="np.exp", func=np.exp)
register_operation(name=".idxmax", func=lambda d, *a, **k: d.idxmax(*a, **k))
register_operation(name=".stack", func=lambda d, *a, **k: d.stack(*a, **k))


def apply_along_dim(func):
    """Decorator which allows for applying a function, which acts on an
    array-like, along a dimension of a xarray.DataArray.
    """

    def _apply_along_axis(data, along_dim: str = None, *args, **kwargs):
        if along_dim is not None:
            dset_collection = {'param': along_dim, 'data': {}}
            for idx in range(len(data[along_dim])):
                dset_collection['data'][str(data[along_dim][idx].item())] = func(data.isel({along_dim: idx}),
                                                                                 *args,
                                                                                 **kwargs)
            return dset_collection
        else:
            return func(data, *args, **kwargs)

    return _apply_along_axis


@is_operation('HarrisWilson.get_marginals')
@apply_along_dim
def get_max_loss_by_bin(data, *, bins: int = 100):
    """ Sorts the data into bins and calculates marginal densities by summing over each bin entry. """

    # Collect all points into a list of tuples and sort by their x value
    zipped_pairs = sorted(
        np.array(
            [z for z in list(zip(data['param1'].values.flatten(), data['loss'].values.flatten()))
             if not (np.isnan(z[0]) or np.isnan(z[1]))]
        ), key=lambda obj: obj[0])

    # Create bins
    x, y = np.linspace(zipped_pairs[0][0], zipped_pairs[-1][0], bins), np.zeros(bins)

    bin_no = 1
    # Sort x into bins; cumulatively gather y-values
    for point in zipped_pairs:
        while point[0] > x[bin_no]:
            bin_no += 1
        y[bin_no - 1] += point[1]

    # Normalise y to 1
    y /= np.sum(y)

    # Combine into a xr.Dataset
    return xr.Dataset(data_vars=dict(prob=('param1', y)), coords=dict(param1=('param1', x)))
