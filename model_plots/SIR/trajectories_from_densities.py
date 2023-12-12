import logging

import numpy as np
import torch
import xarray as xr
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from models.SIR import generate_smooth_data
from utopya.eval import is_operation

log = logging.getLogger(__name__)


def _adjust_for_time_dependency(
    param_cfg: dict, cfg: dict, true_counts: xr.Dataset
) -> dict:
    """Adjusts the parameter configuration for time dependent parameters, if given."""

    time_dependent_parameters = cfg["Data"].get("time_dependent_parameters", {})

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


@is_operation("SIR_densities_from_joint")
def densities_from_joint(
    joint: xr.Dataset,
    *,
    true_counts: xr.Dataset,
    cfg: dict,
) -> xr.Dataset:
    """Runs the SIR ODE model with the estimated parameters, given in the xr.Dataset, and weights each time series with
    its corresponding probability. The probabilities must be normalised to 1.

    :param joint: the ``xr.Dataset`` of the joint parameter distribution
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data (only needed if parameters are time dependent)
    :return: an ``xr.Dataset`` of the mean, mode, std, and true densities for all compartments
    """

    res = []

    # Stack the parameters into a single coordinate
    joint = joint.stack(sample=list(joint.coords))

    # Remove parameters with probability 0
    joint = joint.where(joint > 0, drop=True)

    # Normalise the probabilities to 1 (this is not the same as integrating over the joint -- we are calculating the
    # expectation value only over the samples we are drawing, not of the entire joint distribution!)
    joint /= joint.sum()

    sample_cfg = cfg["Data"]["synthetic_data"]

    with logging_redirect_tqdm():
        for s in (pbar := trange(len(joint.coords["sample"]))):
            pbar.set_description(
                f"Drawing {len(joint.coords['sample'])} samples from joint distribution: "
            )

            # Get the sample
            sample = joint.isel({"sample": [s]})

            # Construct the configuration, taking time-dependent parameters into account
            sample = sample.unstack("sample")

            sample_cfg.update({key: val.item() for key, val in sample.coords.items()})
            param_cfg = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)

            # Generate smooth data
            generated_data = generate_smooth_data(
                cfg=param_cfg,
                num_steps=len(true_counts.coords["time"]) - 1,
                dt=cfg["Data"]["synthetic_data"].get("dt", None),
                init_state=torch.from_numpy(
                    true_counts.isel({"time": 0}, drop=True).data
                ).float(),
                write_init_state=True,
            ).numpy()

            res.append(
                xr.DataArray(
                    data=[[generated_data]],
                    dims=["sample", "type", "time", "kind", "dim_name__0"],
                    coords=dict(
                        sample=[s],
                        type=["mean prediction"],
                        time=true_counts.coords["time"],
                        kind=true_counts.coords["kind"],
                        dim_name__0=true_counts.coords["dim_name__0"],
                    ),
                )
            )

    # Concatenate all the time series
    res = xr.concat(res, dim="sample").squeeze(["dim_name__0"], drop=True)

    # Get the index of the most likely parameter
    mode = joint.isel({"sample": joint.argmax(dim="sample")})
    sample_cfg.update({key: val.item() for key, val in mode.coords.items()})

    # Perform a run using the mode
    mode_params = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)
    mode_data = generate_smooth_data(
        cfg=mode_params,
        num_steps=len(true_counts.coords["time"]) - 1,
        dt=cfg["Data"]["synthetic_data"].get("dt", None),
        init_state=torch.from_numpy(
            true_counts.isel({"time": 0}, drop=True).data
        ).float(),
        write_init_state=True,
    ).numpy()

    mode_data = xr.DataArray(
        data=[mode_data],
        dims=["type", "time", "kind", "dim_name__0"],
        coords=dict(
            type=["mode prediction"],
            time=true_counts.coords["time"],
            kind=true_counts.coords["kind"],
            dim_name__0=[0],
        ),
    ).squeeze(["dim_name__0"], drop=True)

    # Reshape the probability array
    prob = np.reshape(joint.data, (len(joint.coords["sample"]), 1, 1, 1))
    prob_stacked = np.repeat(prob, 1, 1)
    prob_stacked = np.repeat(prob_stacked, len(true_counts.coords["time"]), 2)
    prob_stacked = np.repeat(prob_stacked, len(true_counts.coords["kind"]), 3)

    # Calculate the mean and standard deviation by multiplying the predictions with the associated probability
    mean = (res * prob_stacked).sum("sample")
    std = np.sqrt(((res - mean) ** 2 * prob).sum("sample"))

    # Add a type to the true counts to allow concatenation
    true_counts = true_counts.expand_dims({"type": ["true data"]}).squeeze(
        "dim_name__0", drop=True
    )

    mean = xr.concat([mean, true_counts, mode_data], dim="type")
    std = xr.concat([std, 0 * true_counts, 0 * mode_data], dim="type")

    data = xr.Dataset(data_vars=dict(mean=mean, std=std))

    return data
