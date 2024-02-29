import sys

import scipy.integrate

from .data_ops import *


# ----------------------------------------------------------------------------------------------------------------------
# ADJACENCY MATRIX OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("triangles")
@apply_along_dim
def triangles(
    A: xr.DataArray,
    *args,
    input_core_dims: list = ["j"],
    offset=0,
    axis1=1,
    axis2=2,
    directed: bool = True,
    **kwargs,
) -> xr.DataArray:
    """Calculates the number of triangles on each node from an adjacency matrix A along one dimension.
    The number of triangles are given by

        t(i) = sum_{jk} A_{ij} A_{jk} A_{ki}

    in the directed case, which is simply the i-th entry of the diagonal of A**3. If the network is directed,
    the number of triangles must be divided by 2. It is recommended to use ``xr.apply_ufunc`` for the inner
    (the sample) dimension, as the ``apply_along_dim`` decorator is quite slow.

    :param A: the adjacency matrix
    :param offset: (optional) passed to ``np.diagonal``. Offset of the diagonal from the main diagonal.
        Can be positive or negative. Defaults to main diagonal (0).
    :param axis1: (optional) passed to ``np.diagonal``. Axis to be used as the first axis of the
        2-D sub-arrays from which the diagonals should be taken. Defaults to first axis (0).
    :param axis2: (optional) passed to ``np.diagonal``. Axis to be used as the second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults to second axis (1).
    :param input_core_dims: passed to ``xr.apply_ufunc``
    :param directed: (optional, bool) whether the network is directed. If not, the number of triangle on each node
        is divided by 2.
    :param args, kwargs: additional args and kwargs passed to ``np.linalg.matrix_power``
    """

    res = xr.apply_ufunc(
        np.diagonal,
        xr.apply_ufunc(np.linalg.matrix_power, A, 3, *args, **kwargs),
        offset,
        axis1,
        axis2,
        input_core_dims=[input_core_dims, [], [], []],
    )

    if not directed:
        res /= 2

    return res.rename("triangles")


@is_operation("binned_nw_statistic")
@apply_along_dim
def binned_nw_statistic(
    nw_statistic: xr.DataArray,
    *,
    bins: Any,
    ranges: Sequence = None,
    normalize: Union[bool, float] = False,
    sample_dim: str = "batch",
    **kwargs,
) -> xr.DataArray:
    """Calculates a binned statistic from an adjacency matrix statistic along the batch dimension. This function uses
    the `hist_1D` function to speed up computation. Since network statistics are binned along common x-values for each
    prediction element, the x-coordinate is written as a coordinate, rather than a variable like in other marginal
    calculations.

    :param nw_statistic: the xr.DataArray of adjacency matrix statistics (e.g. the degrees), indexed by 'batch'
    :param bins: bins to use. Any argument admitted by `np.histogram` is permissible
    :param ranges: (float, float), optional: range of the bins to use. Defaults to the minimum and maximum value
        along *all* predictions.
    :param normalize: whether to normalize bin counts.
    :param sample_dim: name of the sampling dimension, which will be excluded from histogramming
    :param kwargs: passed to ``hist``
    :return: xr.Dataset of binned statistics, indexed by the batch index and x-value
    """

    _along_dim = list(nw_statistic.coords)
    _along_dim.remove(sample_dim)
    return hist(
        nw_statistic,
        bins=bins,
        ranges=ranges,
        dim=_along_dim[0],
        normalize=normalize,
        use_bins_as_coords=True,
        **kwargs,
    ).rename("y")


@is_operation("sel_matrix_indices")
@apply_along_dim
def sel_matrix_indices(
    A: xr.DataArray, indices: xr.Dataset, drop: bool = False
) -> xr.DataArray:
    """Selects entries from an adjacency matrix A given in ``indices``. If specified, coordinate labels
    are dropped.

    :param A: adjacency matrix with rows and columns labelled ``i`` and ``j``
    :param indices: ``xr.Dataset`` of indices to dropped; variables should be ``i`` and ``j``
    :param drop: whether to drop the ``i`` and ``j`` coordinate labels
    :return: selected entries of ``A``
    """

    A = A.isel(i=(indices["i"]), j=(indices["j"]))
    return A.drop_vars(["i", "j"]) if drop else A


@is_operation("largest_entry_indices")
@apply_along_dim
def largest_entry_indices(
    A: xr.DataArray, n: int, *, symmetric: bool = False
) -> xr.Dataset:
    """Returns the two-dimensional indices of the n largest entries in an adjacency matrix as well as the corresponding
    values. If the matrix is symmetric, only the upper triangle is considered. The entries are returned sorted from
    highest to lowest.

    :param A: adjacency matrix
    :param n: number of entries to obtain
    :param symmetric: (optional) whether the adjacency matrix is symmetric
    :return: ``xr.Dataset`` of largest entries and their indices
    """

    if symmetric:
        indices_i, indices_j = np.unravel_index(
            np.argsort(np.triu(A).ravel()), np.shape(A)
        )
    else:
        indices_i, indices_j = np.unravel_index(np.argsort(A.data.ravel()), np.shape(A))

    i, j = indices_i[-n:][::-1], indices_j[-n:][::-1]
    vals = A.data[i, j]

    return xr.Dataset(
        data_vars=dict(i=("idx", i), j=("idx", j), value=("idx", vals)),
        coords=dict(idx=("idx", np.arange(len(i)))),
    )


# ----------------------------------------------------------------------------------------------------------------------
# DISTRIBUTION OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------


@is_operation("marginal_distribution")
@apply_along_dim
def marginal_distribution(
    predictions: xr.DataArray,
    probabilities: xr.DataArray,
    true_values: xr.DataArray = None,
    *,
    bin_coord: str = "x",
    y: str = "MLE",
    yerr: str = "std",
    **kwargs,
) -> xr.Dataset:
    """Calculates the marginal distribution from a dataset of binned network statistic (e.g. degree distributions).
    The joint of the statistics and the loss is calculated, the marginal over the loss returned, with a y and yerr value
    calculated. The y value can either be the mean or the mode distribution, and the yerr value is the standard deviation
    of the marginal over the loss on each statistic bin. If passed, the true distribution is also appended to the dataset.

    :param predictions: 2D ``xr.DataArray`` of predictions; indexed by sample dimension and bin dimension
    :param probabilities: 1D ``xr.DataArray`` of probabilities, indexed by sample dimension
    :param true_values: (optional) 1D ``xr.DataArray`` of true distributions, indexed by bin dimension
    :param bin_coord: (optional) name of the x-dimension; default is 'x'
    :param y: statistic to calculate for the y variable; default is the maximum likelihood estimator, can also be the
        ``mean``
    :param yerr: error statistic to use for the y variable; default is the standard deviation (std), but can also be
        the interquartile range (iqr)
    :param kwargs: kwargs, passed to ``marginal_from_ds``
    :return: ``xr.Dataset`` of y and yerr values as variables, and x-values as coordinates. If the true values are
        passed, also contains a ``type`` dimension.
    """

    # Temporarily rename the 'x' dimension to avoid potential naming conflicts with the marginal operation,
    # which also produces 'x' values. This is only strictly necessary if the x-dimension is called 'x'.
    predictions = predictions.rename({bin_coord: f"_{bin_coord}"})

    # Broadcast the predictions and probabilities together
    predictions_and_loss = broadcast(predictions, probabilities, x="y", p="prob")

    # Calculate the distribution marginal for each bin
    marginals = marginal_from_ds(
        predictions_and_loss, x="y", y="prob", exclude_dim=[f"_{bin_coord}"], **kwargs
    )

    # Calculate the y-statistic: mode (default) or mean
    if y == "mode" or y == "MLE":
        p_max = probabilities.idxmax()
        _y_vals = predictions.sel({p_max.name: p_max.data}, drop=True)
    elif y == "mean":
        _y_vals = mean(marginals, along_dim=["bin_idx"], x="x", y="y")["mean"]

    # Calculate the standard deviation from y
    _y_err_vals: xr.DataArray = stat_function(
        marginals, along_dim=["bin_idx"], x="x", y="y", stat=yerr
    )[yerr]

    # Interquartile range is total range, so divide by 2, since errorbands are shown as ± err
    if yerr == "iqr":
        _y_err_vals /= 2

    # Combine y and yerr values into a single dataset and rename the 'x' dimension
    res = xr.Dataset(dict(y=_y_vals, yerr=_y_err_vals)).rename({f"_{bin_coord}": bin_coord})

    # If the true values were given, add to the dataset. The true values naturally have zero error.
    if true_values is not None:
        # Assign the x coordinates from res to ensure compatibility, they should be the same anyway
        # but might be different because of precision errors
        true_values = xr.Dataset(
            dict(y=true_values, yerr=0 * true_values)
        ).assign_coords({bin_coord: res.coords[bin_coord]})
        res = concat([res, true_values], "type", [y, "True values"])

    return res


@is_operation("marginal_distribution_stats")
@apply_along_dim
def marginal_distribution_stats(
    predictions: xr.DataArray,
    probabilities: xr.DataArray,
    *,
    distance_to: str = None,
    stat: Sequence,
    **kwargs,
) -> xr.DataArray:
    """Calculates the statistics of a marginal distribution. This operation circumvents having to first compile
    marginals for all dimensions when sweeping, only to then apply a statistics function along the bin dimension,
    thereby saving memory.

    The ``std`` and ``Hellinger`` and ``KL`` error statistics require different marginalisations: the first requires
    marginalising over the probability, while the second and third require marginalising over the counts. This is
    because the ``Hellinger`` and ``KL`` divergences require the probability bins to line up, i.e. to represent the
    same predicted distribution, so that the distance to a target distribution can be computed.

    :param ds: dataset containing x and y variables for which to calculate the marginal
    :param bins: bins to use for the marginal
    :param ranges: ranges to use for the marginal
    :param x: x dimension
    :param y: function values p(x)
    :param stats: list or string of statistics to calculate. Can be any argument accepted by ``_stat_function``, or
        ``mode``, ``Hellinger``, or ``KL``.
    :param kwargs: additional kwargs, passed to the marginal function
    :return: xr.Dataset of marginal statistics
    """

    stat = set(stat)
    if "Hellinger" in stat or "KL" in stat:
        if distance_to is None:
            raise ValueError(
                f"Calculating Hellinger or relative entropy statistics requires the 'distance_to' kwarg!"
            )

    # Temporarily rename the 'x' dimension to avoid naming conflicts with the marginal operation,
    # which also produces 'x' values
    predictions = predictions.rename({"x": "_x"})

    # Broadcast the predictions and probabilities together, and drop any distributions that are completely zero
    predictions_and_loss = broadcast(predictions, probabilities, x="y", p="prob")
    predictions_and_loss = predictions_and_loss.where(
        predictions_and_loss["prob"] > 0, drop=True
    )

    # Calculate the distribution marginal for each bin. These are different for the Hellinger and KL divergences,
    # since these require the marginal coordinates to align for each _x value, since they must represent one
    # single distribution.
    if stat != {"KL", "Hellinger"} or (
        ("Hellinger" in stat or "KL" in stat) and "distance_to" == "mean"
    ):
        marginal_over_p = marginal_from_ds(
            predictions_and_loss, x="y", y="prob", exclude_dim=["_x"], **kwargs
        )

    if "Hellinger" in stat or "KL" in stat:
        # For Hellinger and KL statistics, marginalise over the counts dimension
        marginal_over_counts = marginal_from_ds(
            predictions_and_loss,
            x="prob",
            y="y",
            exclude_dim=["_x"],
            normalize=False,
            **kwargs,
        )

        # Get the Q distribution with respect to which the error is to be calculated
        if distance_to == "mode" or distance_to == "MLE":
            _y_vals = marginal_over_counts["y"].isel({"bin_idx": -1}, drop=True)
        elif distance_to == "mean":
            _y_vals = mean(marginal_over_p, along_dim=["bin_idx"], x="x", y="y")["mean"]

        # Get the binned loss values associated with each marginal entry
        prob_binned = marginal_over_counts["x"].isel({"_x": 0}, drop=True)

    # Calculate all required statistics
    res = []

    # Calculate the standard deviation from y
    for _stat in stat:
        # Average Hellinger distance
        if _stat == "Hellinger":
            _distributions = Hellinger_distance(
                marginal_over_counts["y"],
                _y_vals.expand_dims(
                    {"bin_idx": marginal_over_counts.coords["bin_idx"]}
                ),
                exclude_dim=["bin_idx"],
            )
            _err = (
                prob_binned * _distributions["Hellinger_distance"] / prob_binned.sum()
            ).sum("bin_idx")
            res.append(
                xr.DataArray(_err.data, name="stat").expand_dims({"type": [_stat]})
            )

        # Average relative entropy
        elif _stat == "KL":
            _distributions = relative_entropy(
                marginal_over_counts["y"],
                _y_vals.expand_dims(
                    {"bin_idx": marginal_over_counts.coords["bin_idx"]}
                ),
                exclude_dim=["bin_idx"],
            )

            _err = (
                prob_binned
                * abs(_distributions["relative_entropy"])
                / prob_binned.sum("bin_idx")
            ).sum("bin_idx")
            res.append(
                xr.DataArray(_err.data, name="stat").expand_dims({"type": [_stat]})
            )
        else:
            # Integrate the standard deviation along x
            _err = stat_function(
                marginal_over_p, along_dim=["bin_idx"], x="x", y="y", stat=_stat
            )[_stat]
            res.append(
                xr.DataArray(
                    scipy.integrate.trapezoid(_err.data, _err.coords["_x"]), name="stat"
                ).expand_dims({"type": [_stat]})
            )

    return xr.concat(res, "type")
