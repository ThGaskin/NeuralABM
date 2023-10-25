import sys

import scipy.integrate

from .data_ops import *


# ----------------------------------------------------------------------------------------------------------------------
# ADJACENCY MATRIX OPERATIONS
# ----------------------------------------------------------------------------------------------------------------------
@is_operation("triangles_1D")
def triangles_1D(
    ds: xr.DataArray,
    offset=0,
    axis1=1,
    axis2=2,
    *args,
    input_core_dims: list = ["j"],
    **kwargs,
) -> xr.DataArray:
    """Calculates the number of triangles on each node from an adjacency matrix along one dimension.
    The number of triangles are given by

        t(i) = 1/2 sum a_{ij}_a{jk}a_{ki}

    in the undirected case, which is simply the i-th entry of the diagonal of A**3. This does not use the apply_along_dim
    function is thus fast, but cannot be applied along multiple dimensions.

    :param a: the adjacency matrix
    :param offset, axis1, axis2: passed to `np.diagonal`
    :param input_core_dims: passed to `xr.apply_ufunc`
    :param args, kwargs: additional args and kwargs passed to `np.linalg.matrix_power`

    """

    res = xr.apply_ufunc(np.linalg.matrix_power, ds, 3, *args, **kwargs)
    res = xr.apply_ufunc(
        np.diagonal,
        res,
        offset,
        axis1,
        axis2,
        input_core_dims=[input_core_dims, [], [], []],
    )
    coords = dict(ds.coords)
    coords.pop(input_core_dims[0])
    dims = list(ds.sizes.keys())
    dims.remove(input_core_dims[0])

    return xr.DataArray(res, dims=dims, coords=coords, name="triangles")


@is_operation("triangles")
@apply_along_dim
def triangles(
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


@is_operation("binned_nw_statistic")
@apply_along_dim
def binned_nw_statistic(
    nw_statistic: xr.DataArray,
    *,
    bins: Any,
    ranges: Sequence = None,
    normalize: bool = True,
) -> xr.Dataset:
    """Calculates a binned statistic from an adjacency matrix statistic along the batch dimension. This function uses
    the `hist_1D` function to speed up computation. Since network statistics are binned along common x-values for each
    prediction element, the x-coordinate is written as a coordinate, rather than a variable like in other marginal
    calculations.

    :param nw_statistic: the xr.DataArray of adjacency matrix statistics (e.g. the degrees), indexed by 'batch'
    :param bins: bins to use. Any argument admitted by `np.histogram` is permissible
    :param ranges: (float, float), optional: range of the bins to use. Defaults to the minimum and maximum value
        along *all* predictions.
    :param normalize: whether to normalize bin counts.
    :return: xr.Dataset of binned statistics, indexed by the batch index and x-value
    """

    _along_dim = list(nw_statistic.coords)
    _along_dim.remove("batch")
    return hist_1D(
        nw_statistic, bins, ranges, along_dim=_along_dim, normalize=normalize
    ).rename({nw_statistic.name: "y"})


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
    :param y: statistic to calculate for the y variable; default is the maximum likelihood estimator, can also be the
        ``mean``
    :param yerr: error statistic to use for the y variable; default is the standard deviation (std), but can also be
        the interquartile range (iqr)
    :param kwargs: kwargs, passed to ``marginal_from_ds``
    :return: ``xr.Dataset`` of y and yerr values as variables, and x-values as coordinates. If the true values are
        passed, also contains a ``type`` dimension.
    """

    # Temporarily rename the 'x' dimension to avoid naming conflicts with the marginal operation,
    # which also produces 'x' values
    predictions = predictions.rename({"x": "_x"})

    # Broadcast the predictions and probabilities together
    predictions_and_loss = broadcast(predictions["y"], probabilities, x="y", p="prob")

    # Calculate the distribution marginal for each bin
    marginals = marginal_from_ds(
        predictions_and_loss, x="y", y="prob", exclude_dim=["_x"], **kwargs
    )

    # Calculate the y-statistic: mode (default) or mean
    if y == "mode" or y == "MLE":
        p_max = probabilities.idxmax()
        _y_vals = predictions.sel({p_max.name: p_max.data}, drop=True)["y"]
    elif y == "mean":
        _y_vals = mean(marginals, along_dim=["bin_idx"], x="x", p="marginal")["mean"]

    # Calculate the standard deviation from y
    _y_err_vals = stat_function(
        marginals, along_dim=["bin_idx"], x="x", p="marginal", stat=yerr
    )[yerr]

    # Combine y and yerr values into a single dataset and rename the 'x' dimension
    res = xr.Dataset(dict(y=_y_vals, yerr=_y_err_vals)).rename({"_x": "x"})

    # If the true values were given, add to the dataset. The true values naturally have zero error.
    if true_values is not None:
        # Assign the x coordinates from res to ensure compatibility, they should be the same anyway
        # but might be different because of precision errors
        true_values = xr.Dataset(
            dict(y=true_values, yerr=0 * true_values)
        ).assign_coords({"x": res.coords["x"]})
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
) -> xr.Dataset:
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
    predictions_and_loss = broadcast(predictions["y"], probabilities, x="y", p="prob")
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
            _y_vals = marginal_over_counts["marginal"].isel({"bin_idx": -1}, drop=True)
        elif distance_to == "mean":
            _y_vals = mean(marginal_over_p, along_dim=["bin_idx"], x="x", p="marginal")[
                "mean"
            ]

        # Get the binned loss values associated with each marginal entry
        prob_binned = marginal_over_counts["x"].isel({"_x": 0}, drop=True)

    # Calculate all required statistics
    res = []

    # Calculate the standard deviation from y
    for _stat in stat:
        # Average Hellinger distance
        if _stat == "Hellinger":
            _distributions = Hellinger_distance(
                marginal_over_counts["marginal"],
                _y_vals.expand_dims(
                    {"bin_idx": marginal_over_counts.coords["bin_idx"]}
                ),
                exclude_dim=["bin_idx"],
            )
            _err = (
                prob_binned * _distributions["Hellinger_distance"] / prob_binned.sum()
            ).sum("bin_idx")
            res.append(xr.DataArray(_err.data, name="stat"))

        # Average relative entropy
        elif _stat == "KL":
            _distributions = relative_entropy(
                marginal_over_counts["marginal"],
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
            res.append(xr.DataArray(_err.data, name="stat"))
        else:
            # Integrate the standard deviation along x
            _err = stat_function(
                marginal_over_p, along_dim=["bin_idx"], x="x", p="marginal", stat=_stat
            )[_stat]
            res.append(
                xr.DataArray(
                    scipy.integrate.trapezoid(_err.data, _err.coords["_x"]), name="stat"
                )
            )

    return concat(res, "type", stat)


# ----------------------------------------------------------------------------------------------------------------------
# MATRIX SELECTION OPERATIONS
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
    If the matrix is symmetric, only the upper triangle is considered. Sorted from highest to lowest.
    """

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
