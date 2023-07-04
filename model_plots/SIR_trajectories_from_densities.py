import logging

import numpy as np
import torch
import xarray as xr

import models.SIR
from models.SIR.DataGeneration import generate_smooth_data
from utopya.eval import is_operation

log = logging.getLogger(__name__)


def _draw_from_marginal(marginal: xr.Dataset) -> float:
    """Draw a single parameter from a 1-d marginal, given by a Dataset of x-values and
    associated probabilities"""

    s = np.random.rand()
    dx = marginal["x"].isel({"bin_idx": 1}) - marginal["x"].isel({"bin_idx": 0})
    p_total, i = 0, 0
    while p_total < s:
        p_total += marginal["p"].isel({"bin_idx": i}) * dx
        i += 1
    return marginal["x"].isel({"bin_idx": i}).item()


def _mode_from_marginal(marginal: xr.Dataset) -> float:
    """Returns the mode of the marginal"""
    argmax = marginal["p"].argmax()
    return marginal["x"].isel({"bin_idx": argmax}).item()


def _get_sample(marginals: xr.Dataset):
    """Returns an n-dimensional vector of samples for each marginal distribution."""
    vec = {}
    for param in marginals.coords["parameter"]:
        vec[param.item()] = _draw_from_marginal(marginals.sel({"parameter": param}))
    return vec


def _get_mode(marginals: xr.Dataset):
    """Returns the mode of a marginal"""
    vec = {}
    for param in marginals.coords["parameter"]:
        vec[param.item()] = _mode_from_marginal(marginals.sel({"parameter": param}))
    return vec


def _adjust_for_time_dependency(
    param_cfg: dict, cfg: dict, true_counts: xr.Dataset
) -> dict:

    """Adjusts the parameter configuration for time dependent parameters, if given."""

    time_dependent_parameters = cfg["Data"].get("time_dependent_params", {})

    # Extend any time-dependent parameters, if necessary
    for param in time_dependent_parameters.keys():
        val = np.zeros(len(true_counts.coords["time"]))
        i = 0
        # Replace any time-dependent parameters with a series
        for j, interval in enumerate(time_dependent_parameters[param]):
            _, upper = interval
            if not upper:
                upper = len(val)
            while i < upper:
                val[i] = param_cfg[str(param) + f"_{j}"]
                i += 1
        param_cfg[param] = val

    return param_cfg


def _densities_from_marginals(
    data: xr.Dataset,
    generating_func,
    *,
    num_samples: int,
    true_counts: xr.Dataset,
    cfg: dict,
) -> xr.Dataset:

    """Draws N samples for each parameter from a collection of marginals, then runs the simulation and returns the
    averaged densities, standard deviation, mode, and true counts in a single xr.Dataset, ready to be passed to
    .plot.facet_grid.line.

    :param data: the xr.Dataset of marginals, indexed by the parameter name
    :param generating_func: the generating function used to run the model
    :param num_samples: the number of samples to draw from the marginals
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data
    :return: an xr.Dataset of the mean, mode, std, and true densities for all compartments
    """

    res = []
    for n in range(num_samples):
        parameters = _adjust_for_time_dependency(_get_sample(data), cfg, true_counts)
        generated_data = generating_func(
            cfg=parameters,
            num_steps=len(true_counts.coords["time"]) - 1,
            dt=cfg["Data"]["synthetic_data"].get("dt", None),
            k_q=cfg["Data"]["synthetic_data"].get("k_q", None),
            write_init_state=True,
            init_state=torch.from_numpy(
                true_counts.isel({"time": 0}, drop=True).data
            ).float(),
        ).numpy()
        res.append(
            xr.DataArray(
                data=[np.stack([true_counts, generated_data])],
                dims=["sample", "type", "time", "kind", "dim_name__0"],
                coords=dict(
                    sample=[n],
                    type=["true_counts", "prediction_mean"],
                    time=true_counts.coords["time"],
                    kind=true_counts.coords["kind"],
                    dim_name__0=true_counts.coords["dim_name__0"],
                ),
            )
        )

    res = xr.concat(res, dim="sample")

    # Perform a run using the mode
    mode_params = _adjust_for_time_dependency(_get_mode(data), cfg, true_counts)
    mode_data = generate_smooth_data(
        cfg=mode_params,
        num_steps=len(true_counts.coords["time"]) - 1,
        dt=cfg["Data"]["synthetic_data"].get("dt", None),
        k_q=cfg["Data"]["synthetic_data"].get("k_q", None),
        write_init_state=True,
        init_state=torch.from_numpy(
            true_counts.isel({"time": 0}, drop=True).data
        ).float(),
    ).numpy()

    mode_data = xr.DataArray(
        data=[mode_data],
        dims=["type", "time", "kind", "dim_name__0"],
        coords=dict(
            type=["prediction_mode"],
            time=res.coords["time"],
            kind=res.coords["kind"],
            dim_name__0=res.coords["dim_name__0"],
        ),
    )

    # Calculate the mean and standard deviation
    mean = xr.concat([res.mean("sample"), mode_data], dim="type")
    std = xr.concat([res.std("sample"), 0 * mode_data], dim="type")

    data = xr.Dataset(data_vars=dict(mean=mean, std=std))

    return data


def _densities_from_joint(
    parameters: xr.Dataset,
    prob: xr.Dataset,
    generating_func,
    *,
    true_counts: xr.Dataset,
    cfg: dict,
):

    """Runs the model with the estimated parameters, given in the xr.Dataset, and weights each time series with
    its corresponding probability. The probabilities must be normalised to 1.

    :param data: the xr.Dataset of parameter estimates, indexed by sample
    :param prob: the xr.Dataset of probabilities associated with each estimate, indexed by sample
    :param generating_func: the generating function used to run the model
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data
    :return: an xr.Dataset of the mean, mode, std, and true densities for all compartments
    """

    res = []
    for s in range(len(parameters.coords["sample"])):

        # Construct the configuration, taking time-dependent parameters into account
        sample = parameters.isel({"sample": s}, drop=True)
        sample_cfg = dict(
            (p.item(), sample.sel({"parameter": p}).item())
            for p in sample.coords["parameter"]
        )
        param_cfg = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)

        # Generate smooth data
        generated_data = generating_func(
            cfg=param_cfg,
            num_steps=len(true_counts.coords["time"]) - 1,
            dt=cfg["Data"]["synthetic_data"].get("dt", None),
            k_q=cfg["Data"]["synthetic_data"].get("k_q", None),
            write_init_state=True,
            init_state=torch.from_numpy(
                true_counts.isel({"time": 0}, drop=True).data
            ).float(),
        ).numpy()
        res.append(
            xr.DataArray(
                data=[[generated_data]],
                dims=["sample", "type", "time", "kind", "dim_name__0"],
                coords=dict(
                    sample=[s],
                    type=["prediction_mean"],
                    time=true_counts.coords["time"],
                    kind=true_counts.coords["kind"],
                    dim_name__0=true_counts.coords["dim_name__0"],
                ),
            )
        )

    # Concatenate all the time series
    res = xr.concat(res, dim="sample").squeeze(["dim_name__0"], drop=True)

    # Get the index of the most likely parameter
    mode_idx = prob.argmax(dim="sample")
    mode_cfg = dict(
        (
            p.item(),
            parameters.isel({"sample": mode_idx}, drop=True)
            .sel({"parameter": p})
            .item(),
        )
        for p in parameters.coords["parameter"]
    )

    # Perform a run using the mode
    mode_params = _adjust_for_time_dependency(mode_cfg, cfg, true_counts)
    mode_data = generating_func(
        cfg=mode_params,
        num_steps=len(true_counts.coords["time"]) - 1,
        dt=cfg["Data"]["synthetic_data"].get("dt", None),
        k_q=cfg["Data"]["synthetic_data"].get("k_q", None),
        write_init_state=True,
        init_state=torch.from_numpy(
            true_counts.isel({"time": 0}, drop=True).data
        ).float(),
    ).numpy()

    mode_data = xr.DataArray(
        data=[mode_data],
        dims=["type", "time", "kind", "dim_name__0"],
        coords=dict(
            type=["prediction_mode"],
            time=true_counts.coords["time"],
            kind=true_counts.coords["kind"],
            dim_name__0=[0],
        ),
    ).squeeze(["dim_name__0"], drop=True)

    # Reshape the probability array
    prob = np.reshape(prob.data, (len(prob.coords["sample"]), 1, 1, 1))
    prob_stacked = np.repeat(prob, 1, 1)
    prob_stacked = np.repeat(prob_stacked, len(true_counts.coords["time"]), 2)
    prob_stacked = np.repeat(prob_stacked, len(true_counts.coords["kind"]), 3)

    # Calculate the mean and standard deviation by multiplying the predictions with the associated probability
    mean = (res * prob_stacked).sum("sample")
    std = np.sqrt(((res - mean) ** 2 * prob).sum("sample"))

    # Add a type to the true counts to allow concatenation
    true_counts = true_counts.expand_dims({"type": ["true_counts"]}).squeeze(
        "dim_name__0", drop=True
    )

    mean = xr.concat([mean, true_counts, mode_data], dim="type")
    std = xr.concat([std, 0 * true_counts, 0 * mode_data], dim="type")

    data = xr.Dataset(data_vars=dict(mean=mean, std=std))

    return data


@is_operation("SIR_densities_from_marginals")
def SIR_densities_from_marginals(
    data: xr.Dataset, *, num_samples: int, true_counts: xr.Dataset, cfg: dict
) -> xr.Dataset:
    return _densities_from_marginals(
        data,
        models.SIR.generate_smooth_data,
        num_samples=num_samples,
        true_counts=true_counts,
        cfg=cfg,
    )


@is_operation("SIR_densities_from_joint")
def SIR_densities_from_joint(
    data: xr.Dataset, prob: xr.Dataset, *, true_counts: xr.Dataset, cfg: dict
) -> xr.Dataset:

    return _densities_from_joint(
        data, prob, models.SIR.generate_smooth_data, true_counts=true_counts, cfg=cfg
    )
