import logging
from typing import Union

import numpy as np
import torch
import xarray as xr
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from models.Covid import generate_smooth_data
from utopya.eval import is_operation

log = logging.getLogger(__name__)

from ..SIR.trajectories_from_densities import _adjust_for_time_dependency


def _combine_compartments(
    da: Union[xr.DataArray, xr.Dataset],
    combine: dict,
) -> Union[xr.DataArray, xr.Dataset]:
    """Combines certain compartments by summation"""

    for key, values in combine.items():
        q = da.sel({"kind": values}).sum("kind").expand_dims({"kind": [key]})
        da = xr.concat(([da, q]), dim="kind")
        da = da.drop_sel({"kind": values})

    return da


def _drop_compartments(
    da: Union[xr.DataArray, xr.Dataset],
    drop: list,
) -> Union[xr.DataArray, xr.Dataset]:
    """Drops certain compartments"""
    return da.drop_sel({"kind": drop})


def _calculate_residuals(da: xr.DataArray) -> xr.DataArray:
    """Calculates the residuals between the predictions and the true data"""

    residuals = xr.DataArray(
        data=[
            (
                da.sel({"type": "prediction_mode"}, drop=True)
                - da.sel({"type": "true_counts"}, drop=True)
            )
            / da.sel({"type": "true_counts"}, drop=True),
            (
                da.sel({"type": "prediction_mean"}, drop=True)
                - da.sel({"type": "true_counts"}, drop=True)
            )
            / da.sel({"type": "true_counts"}, drop=True),
        ],
        dims=["type", "time", "kind"],
        coords=dict(
            type=["mode_residual", "mean_residual"],
            kind=da.coords["kind"],
            time=da.coords["time"],
        ),
    )

    return xr.where(residuals != np.inf, residuals, np.nan)


@is_operation("Covid_densities_residuals")
def print_residuals(data, train_cut: int = 200):
    """Prints a summary of the residuals for each compartment"""

    residuals = _calculate_residuals(data) ** 2

    print_intervals = {"train": slice(None, train_cut), "test": slice(train_cut, None)}

    for key, item in print_intervals.items():
        log.remark("------------------------------------------------------")
        log.remark(f"L2 residuals in {key}")
        l2_mean = np.sqrt(residuals.isel({"time": item}).mean("time", skipna=True))
        for k in residuals.coords["kind"]:
            log.remark(
                f"   {k.item().capitalize()}: {np.around(l2_mean.sel({'kind': k, 'type': 'mean_residual'}).data.item(), 5)} (mean), {np.around(l2_mean.sel({'kind': k, 'type': 'mode_residual'}).data.item(), 5)} (mode)"
            )
        log.remark(
            f"   Average: {np.around(l2_mean.sel({'type': 'mean_residual'}).mean('kind').data.item(), 5)} (mean), {np.around(l2_mean.sel({'type': 'mode_residual'}).mean('kind').data.item(), 5)} (mode)"
        )

    return data


@is_operation("Covid_densities_from_joint")
def densities_from_joint(
    parameters: xr.Dataset,
    prob: xr.Dataset,
    *,
    true_counts: xr.Dataset,
    cfg: dict,
    combine: dict = None,
    drop: list = None,
    mean: xr.Dataset = None,
) -> xr.Dataset:
    """Runs the model with the estimated parameters, given in an ``xr.Dataset`` ``parameters``,
    and weights each time series with its corresponding probability, given ``prob``.
    The probabilities must be normalised to 1.

    :param parameters: the ``xr.Dataset`` of parameter estimates, indexed by the sample dimension
    :param prob: the xr.Dataset of probabilities associated with each estimate, indexed by sample
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data
    :param combine: (optional) dictionary of compartments to combine by summing
    :param drop: (optional) list of compartments to drop
    :param print_residuals: (optional) whether to print the L2 residuals on the test and training periods
    :param train_cut: (optional) time step separating the training from the test period
        (for printing the residuals only)
    :param mean: (optional) can pass the mean dataset instead of calculating it; needed for MCMC calculationss
    :return: an ``xr.Dataset`` of the mean, mode, std, and true densities (if given) for all compartments
    """

    # Name of sample dimension
    sample_dim: str = list(prob.coords.keys())[0]

    # Sample configuration
    sample_cfg = cfg["Data"]["synthetic_data"]
    sample_cfg["num_steps"] = len(true_counts.coords["time"])
    sample_cfg["burn_in"] = 0
    res = []

    with logging_redirect_tqdm():
        for s in (pbar := trange(len(parameters.coords[sample_dim]))):
            pbar.set_description(
                f"Drawing {len(parameters.coords[sample_dim])} samples from joint distribution: "
            )

            # Construct the configuration, taking time-dependent parameters into account
            sample = parameters.isel({sample_dim: s}, drop=True)
            sample_cfg.update(
                {
                    p.item(): sample.sel({"parameter": p}).item()
                    for p in sample.coords["parameter"]
                }
            )
            param_cfg = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)

            # Generate smooth data
            generated_data = generate_smooth_data(
                cfg=param_cfg,
                init_state=torch.from_numpy(
                    true_counts.isel({"time": 0}, drop=True).data
                ).float(),
            ).numpy()

            res.append(
                xr.DataArray(
                    data=[[generated_data]],
                    dims=[sample_dim, "type", "time", "kind", "dim_name__0"],
                    coords=dict(
                        **{sample_dim: [s]},
                        type=["prediction_mean"],
                        **true_counts.coords,
                    ),
                ).squeeze(["dim_name__0"], drop=True)
            )

    # Concatenate all the time series
    res = xr.concat(res, dim=sample_dim)

    # Get the index of the most likely parameter
    mode_idx = prob.argmax(dim=sample_dim)
    sample_cfg.update(
        {
            p.item(): parameters.isel({sample_dim: mode_idx}, drop=True)
            .sel({"parameter": p})
            .item()
            for p in parameters.coords["parameter"]
        }
    )

    # Perform a run using the mode
    mode_params = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)
    mode_data = generate_smooth_data(
        cfg=mode_params,
        init_state=torch.from_numpy(
            true_counts.isel({"time": 0}, drop=True).data
        ).float(),
    ).numpy()

    mode_data = xr.DataArray(
        data=[mode_data],
        dims=["type", "time", "kind", "dim_name__0"],
        coords=dict(type=["prediction_mode"], **true_counts.coords),
    ).squeeze(["dim_name__0"], drop=True)

    # Combine compartments, if given
    if combine:
        res = _combine_compartments(res, combine)
        mode_data = _combine_compartments(mode_data, combine)
        true_counts = _combine_compartments(true_counts, combine)

    # Drop compartments, if given
    if drop:
        res = _drop_compartments(res, drop)
        mode_data = _drop_compartments(mode_data, drop)
        true_counts = _drop_compartments(true_counts, drop)

    # Reshape the probability array
    prob = np.reshape(prob.data, (len(prob.coords[sample_dim]), 1, 1, 1))
    prob_stacked = np.repeat(prob, 1, 1)
    prob_stacked = np.repeat(prob_stacked, len(res.coords["time"]), 2)
    prob_stacked = np.repeat(prob_stacked, len(res.coords["kind"]), 3)

    # Calculate the mean and standard deviation by multiplying the predictions with the associated probability
    mean = (res * prob_stacked).sum(sample_dim) if mean is None else mean
    std = np.sqrt(((res - mean) ** 2 * prob).sum(sample_dim))

    # Add a type to the true counts to allow concatenation
    true_counts = true_counts.expand_dims({"type": ["true_counts"]}).squeeze(
        "dim_name__0", drop=True
    )

    mean = xr.concat([mean, true_counts, mode_data], dim="type")
    std = xr.concat([std, 0 * true_counts, 0 * mode_data], dim="type")

    return xr.Dataset(data_vars=dict(mean=mean, std=std))
