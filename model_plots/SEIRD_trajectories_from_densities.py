import logging
from typing import Union

import numpy as np
import torch
import xarray as xr

import models.SEIRD
import models.SIR
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


def _get_residuals(da: xr.DataArray) -> xr.DataArray:

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


def _densities_from_marginals(
    data: xr.Dataset,
    generating_func,
    *,
    num_samples: int,
    num_steps: int = None,
    dt: float = None,
    true_counts: xr.Dataset,
    cfg: dict,
    combine: dict = None,
    drop: list = None,
    append_true_counts: bool = True,
) -> xr.Dataset:

    """Draws N samples for each parameter from a collection of marginals, then runs the simulation and returns the
    averaged densities, standard deviation, mode, and (optionally) the true counts in a single xr.Dataset, ready to be passed to
    .plot.facet_grid.line.

    :param data: the xr.Dataset of marginals, indexed by the parameter name
    :param generating_func: the generating function used to run the model
    :param num_samples: the number of samples to draw from the marginals
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data
    :param combine: dictionary of compartments to combine by summation into a new compartment,
        keyed by the name of the new compartment
    :param drop: list of compartments to drop
    :param append_true_counts: whether to append the true counts to the resulting dataset
    :return: an xr.Dataset of the mean, mode, std, and true densities for all compartments
    """

    res = []

    # Get the number of steps for which to run the numerical solver
    num_steps = len(true_counts.coords["time"]) - 1 if num_steps is None else num_steps

    for n in range(num_samples):
        parameters = _adjust_for_time_dependency(_get_sample(data), cfg, true_counts)
        generated_data = generating_func(
            cfg=parameters,
            num_steps=num_steps,
            dt=cfg["Data"]["synthetic_data"].get("dt", None) if dt is None else dt,
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
                    sample=[n],
                    type=["prediction_mean"],
                    time=np.arange(num_steps),
                    kind=true_counts.coords["kind"],
                    dim_name__0=true_counts.coords["dim_name__0"],
                ),
            )
        )

    res = xr.concat(res, dim="sample")

    # Perform a run using the mode
    mode_params = _adjust_for_time_dependency(_get_mode(data), cfg, true_counts)
    mode_data = generating_func(
        cfg=mode_params,
        num_steps=len(res.coords["data"]),
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

    # Calculate the mean and standard deviation, and append the true data, if given
    if append_true_counts:
        mean = xr.concat([res.mean("sample"), mode_data, true_counts], dim="type")
        std = xr.concat([res.std("sample"), 0 * mode_data, 0 * true_counts], dim="type")
    else:
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
    num_steps: int = None,
    dt: float = None,
    cfg: dict,
    combine: dict = None,
    drop: list = None,
    append_true_counts: bool = True,
    print_intervals: dict = None,
    mean: xr.Dataset = None,
):

    """Runs the model with the estimated parameters, given in the xr.Dataset, and weights each time series with
    its corresponding probability. The probabilities must be normalised to 1.

    :param data: the xr.Dataset of parameter estimates, indexed by sample
    :param prob: the xr.Dataset of probabilities associated with each estimate, indexed by sample
    :param generating_func: the generating function used to run the model
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data
    :param combine: dictionary of compartments to combine by summation into a new compartment,
        keyed by the name of the new compartment
    :param drop: list of compartments to drop
    :param append_true_counts: whether to append the true counts to the resulting dataset
    :return: an xr.Dataset of the mean, mode, std, and true densities (if given) for all compartments
    """

    res = []

    # Get the number of steps for which to run the numerical solver
    num_steps = len(true_counts.coords["time"]) - 1 if num_steps is None else num_steps

    sample_cfg = cfg["Data"]["synthetic_data"]

    for s in range(len(parameters.coords["sample"])):

        # Construct the configuration, taking time-dependent parameters into account
        sample = parameters.isel({"sample": s}, drop=True)
        sample_cfg.update(
            {
                p.item(): sample.sel({"parameter": p}).item()
                for p in sample.coords["parameter"]
            }
        )
        param_cfg = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)

        # Generate smooth data
        generated_data = generating_func(
            cfg=param_cfg,
            num_steps=num_steps,
            dt=cfg["Data"]["synthetic_data"].get("dt", None) if dt is None else dt,
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
                    time=np.arange(num_steps + 1),
                    kind=true_counts.coords["kind"],
                    dim_name__0=true_counts.coords["dim_name__0"],
                ),
            ).squeeze(["dim_name__0"], drop=True)
        )

    # Concatenate all the time series
    res = xr.concat(res, dim="sample")

    # Get the index of the most likely parameter
    mode_idx = prob.argmax(dim="sample")
    sample_cfg.update(
        {
            p.item(): parameters.isel({"sample": mode_idx}, drop=True)
            .sel({"parameter": p})
            .item()
            for p in parameters.coords["parameter"]
        }
    )

    # Perform a run using the mode
    mode_params = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)
    mode_data = generating_func(
        cfg=mode_params,
        num_steps=num_steps,
        dt=cfg["Data"]["synthetic_data"].get("dt", None) if dt is None else dt,
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
            time=np.arange(num_steps + 1),
            kind=true_counts.coords["kind"],
            dim_name__0=[0],
        ),
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
    # TODO this is not necessary: can be done via broadcasting
    prob = np.reshape(prob.data, (len(prob.coords["sample"]), 1, 1, 1))
    prob_stacked = np.repeat(prob, 1, 1)
    prob_stacked = np.repeat(prob_stacked, len(res.coords["time"]), 2)
    prob_stacked = np.repeat(prob_stacked, len(res.coords["kind"]), 3)

    # Calculate the mean and standard deviation by multiplying the predictions with the associated probability
    mean = (res * prob_stacked).sum("sample") if mean is None else mean
    std = np.sqrt(((res - mean) ** 2 * prob).sum("sample"))

    # Add a type to the true counts to allow concatenation, if required
    true_counts = (
        true_counts.expand_dims({"type": ["true_counts"]})
        .squeeze("dim_name__0", drop=True)
        .isel({"time": slice(0, num_steps, 1)})
    )

    mean = xr.concat([mean, true_counts, mode_data], dim="type")
    std = xr.concat([std, 0 * true_counts, 0 * mode_data], dim="type")

    data = xr.Dataset(data_vars=dict(mean=mean, std=std))

    # Print a summary of the residuals for each compartment
    residuals = _get_residuals(data["mean"]) ** 2

    if print_intervals is None:
        print_intervals = {"train": slice(None, 200), "test": slice(200, None)}
    for key, item in print_intervals.items():
        log.remark("------------------------------------------------------")
        log.remark(f"L2 residuals in {key}")
        l2_mean = np.sqrt(residuals.isel({"time": item}).mean("time", skipna=True))
        for k in residuals.coords["kind"]:
            log.remark(
                f"   {k.item().capitalize()}: {np.around(l2_mean.sel({'kind': k, 'type': 'mean_residual'}).data.item(), 5)} (mean), {np.around(l2_mean.sel({'kind': k, 'type': 'mode_residual'}).data.item(), 5)} (mode)"
            )

    if not append_true_counts:
        return data.sel({"type": ["prediction_mean", "prediction_mode"]}, drop=True)
    else:
        return data


def _densities_from_mode(
    parameters: xr.Dataset,
    prob: xr.Dataset,
    generating_func,
    *,
    true_counts: xr.Dataset,
    cfg: dict,
    num_steps: int = None,
    dt: float = None,
    combine: dict = None,
    drop: list = None,
):

    """Runs the model with the most likely parameters given by the prob xr.Dataset.

    :param data: the xr.Dataset of parameter estimates, indexed by sample
    :param prob: the xr.Dataset of probabilities associated with each estimate, indexed by sample
    :param generating_func: the generating function used to run the model
    :param true_counts: the xr.Dataset of true counts
    :param cfg: the run configuration of the original data
    :param combine: dictionary of compartments to combine by summation into a new compartment,
        keyed by the name of the new compartment
    :param drop: list of compartments to drop
    :return: an xr.Dataset of the mode densities from all compartments
    """

    # Get the index of the most likely parameter
    mode_idx = np.argmax(prob["sample"].data)
    mode_cfg = {
        p.item(): parameters.isel({"sample": mode_idx}, drop=True)
        .sel({"parameter": p})
        .item()
        for p in parameters.coords["parameter"]
    }
    # Perform a run using the mode
    n = len(true_counts.coords["time"]) - 1 if num_steps is None else num_steps
    mode_params = _adjust_for_time_dependency(mode_cfg, cfg, true_counts)
    mode_data = xr.DataArray(
        data=generating_func(
            cfg=mode_params,
            num_steps=n,
            dt=cfg["Data"]["synthetic_data"].get("dt", None) if dt is None else dt,
            k_q=cfg["Data"]["synthetic_data"].get("k_q", None),
            write_init_state=True,
            init_state=torch.from_numpy(
                true_counts.isel({"time": 0}, drop=True).data
            ).float(),
        ).numpy(),
        dims=["time", "kind", "dim_name__0"],
        coords=dict(
            time=np.linspace(0, true_counts.coords["time"][-1], n + 1),
            kind=true_counts.coords["kind"],
            dim_name__0=[0],
        ),
    ).squeeze(["dim_name__0"], drop=True)

    # Combine compartments, if given
    if combine:
        mode_data = _combine_compartments(mode_data, combine)

    # Drop compartments, if given
    if drop:
        mode_data = _drop_compartments(mode_data, drop)

    return mode_data


def _accuracy_and_uncertainty(
    parameters: xr.Dataset,
    loss: xr.Dataset,
    generating_func,
    *,
    sample_step: Union[int, list],
    true_counts: xr.Dataset,
    cfg: dict,
    combine: dict = None,
    drop: list = None,
):
    ts = []

    # Get the number of steps for which to run the numerical solver
    num_steps = len(true_counts.coords["time"]) - 1

    for s in range(len(parameters.coords["sample"])):
        # Construct the configuration, taking time-dependent parameters into account
        sample = parameters.isel({"sample": s}, drop=True)
        sample_cfg = {
            p.item(): sample.sel({"parameter": p}).item()
            for p in sample.coords["parameter"]
        }
        param_cfg = _adjust_for_time_dependency(sample_cfg, cfg, true_counts)

        # Generate smooth data
        generated_data = generating_func(
            cfg=param_cfg,
            num_steps=num_steps,
            dt=cfg["Data"]["synthetic_data"].get("dt", None),
            k_q=cfg["Data"]["synthetic_data"].get("k_q", None),
            write_init_state=True,
            init_state=torch.from_numpy(
                true_counts.isel({"time": 0}, drop=True).data
            ).float(),
        ).numpy()
        ts.append(
            xr.DataArray(
                data=[generated_data],
                dims=["sample", "time", "kind", "dim_name__0"],
                coords=dict(
                    sample=[s],
                    time=np.arange(num_steps + 1),
                    kind=true_counts.coords["kind"],
                    dim_name__0=true_counts.coords["dim_name__0"],
                ),
            ).squeeze(["dim_name__0"], drop=True)
        )

    # Concatenate all the time series
    ts = xr.concat(ts, dim="sample")

    true_counts = true_counts.squeeze(drop=True)
    # Combine and drop compartments
    if combine:
        ts = _combine_compartments(ts, combine)
        true_counts = _combine_compartments(true_counts, combine)
    if drop:
        ts = _drop_compartments(ts, drop)
        true_counts = _drop_compartments(true_counts, drop)

    # For each entry in samples, randomly select a given number of samples from the dataset
    # and calculate the accuracy and uncertainty when using only those samples.
    if isinstance(sample_step, int):
        n_steps = int(len(loss.coords["sample"]) / sample_step)
        samples_list = [
            (n * sample_step, slice(0, n * sample_step)) for n in range(1, n_steps)
        ]
    else:
        samples_list = []
        for n_samples in sample_step:
            if n_samples == -1:
                n_samples = len(loss.coords["sample"])
            samples_list.append(
                (
                    n_samples,
                    np.random.randint(0, ts.coords["sample"][-1], size=n_samples),
                )
            )
    res = []
    for n_samples, sample_idx in samples_list:

        sampled_ts = ts.isel({"sample": sample_idx})
        sampled_loss = loss.isel({"sample": sample_idx})
        sampled_loss /= sampled_loss.sum("sample")

        # Reshape the probability array
        sampled_loss = np.reshape(
            sampled_loss.data, (len(sampled_loss.coords["sample"]), 1, 1)
        )
        sampled_loss = np.repeat(sampled_loss, len(ts.coords["time"]), 1)
        sampled_loss = np.repeat(sampled_loss, len(ts.coords["kind"]), 2)

        # Calculate the mean and standard deviation by multiplying the predictions with the associated probability
        mean = (sampled_ts * sampled_loss).sum("sample")
        std = np.sqrt(((sampled_ts - mean) ** 2 * sampled_loss).sum("sample"))

        residuals = (mean - true_counts) / true_counts
        residuals = np.sqrt(
            np.nanmean(np.where(residuals != np.inf, residuals, np.nan) ** 2, axis=0)
        )

        # Calculate the residuals and average std
        res.append(
            xr.DataArray(
                data=[[residuals, std.mean("time")]],
                dims=["n_samples", "type", "kind"],
                coords=dict(
                    n_samples=[n_samples],
                    type=["residuals", "std"],
                    kind=true_counts.coords["kind"],
                ),
            )
        )

    return xr.concat(res, dim="n_samples")


@is_operation("SIR_generate_smooth_densities")
def SIR_generate_smooth_densities(
    *,
    name: str,
    coords: list = None,
    combine: dict = None,
    drop: list = None,
    cfg: dict,
    num_steps: int,
    init_state: xr.DataArray,
    write_init_state: bool = True,
    **kwargs,
) -> xr.DataArray:

    """Generates smooth SIR data from a configuration."""

    data = models.SIR.generate_smooth_data(
        cfg=cfg,
        num_steps=num_steps,
        write_init_state=write_init_state,
        init_state=torch.from_numpy(init_state.data).float(),
        **kwargs,
    ).numpy()

    da = xr.DataArray(
        data=[data],
        dims=["type", "time", "kind", "dim_name__0"],
        coords=dict(
            type=[name],
            time=np.arange(len(data)),
            kind=coords if coords else ["susceptible", "exposed", "infected"],
            dim_name__0=[0],
        ),
    ).squeeze(["dim_name__0"], drop=True)

    # Combine compartments, if given
    if combine:
        da = _combine_compartments(da, combine)

    # Drop compartments, if given
    if drop:
        da = _drop_compartments(da, drop)

    return da


@is_operation("SEIRD+_generate_smooth_densities")
def SEIRD_generate_smooth_densities(
    *,
    name: str,
    coords: list = None,
    combine: dict = None,
    drop: list = None,
    cfg: dict,
    dt: float = None,
    num_steps: int,
    init_state: xr.DataArray,
    write_init_state: bool = True,
    **kwargs,
):
    data = models.SEIRD.generate_smooth_data(
        cfg=cfg,
        num_steps=num_steps,
        dt=dt,
        write_init_state=write_init_state,
        init_state=torch.from_numpy(init_state.data).float(),
        **kwargs,
    ).numpy()

    da = xr.DataArray(
        data=[data],
        dims=["type", "time", "kind", "dim_name__0"],
        coords=dict(
            type=[name],
            time=np.arange(len(data)),
            kind=coords
            if coords
            else [
                "susceptible",
                "exposed",
                "infected",
                "recovered",
                "symptomatic",
                "hospitalized",
                "critical",
                "deceased",
                "quarantine_S",
                "quarantine_E",
                "quarantine_I",
                "contact_traced",
            ],
            dim_name__0=[0],
        ),
    ).squeeze(["dim_name__0"], drop=True)

    # Combine compartments, if given
    if combine:
        da = _combine_compartments(da, combine)

    # Drop compartments, if given
    if drop:
        da = _drop_compartments(da, drop)

    return da


@is_operation("SIR_densities_from_marginals")
def SIR_densities_from_marginals(
    data: xr.Dataset, *, num_samples: int, true_counts: xr.Dataset, cfg: dict, **kwargs
) -> xr.Dataset:
    return _densities_from_marginals(
        data,
        models.SIR.DataGeneration.generate_smooth_data,
        num_samples=num_samples,
        true_counts=true_counts,
        cfg=cfg,
        **kwargs,
    )


@is_operation("SIR_densities_from_joint")
def SIR_densities_from_joint(
    data: xr.Dataset, prob: xr.Dataset, *, true_counts: xr.Dataset, cfg: dict, **kwargs
) -> xr.Dataset:

    return _densities_from_joint(
        data,
        prob,
        models.SIR.DataGeneration.generate_smooth_data,
        true_counts=true_counts,
        cfg=cfg,
        **kwargs,
    )


@is_operation("SEIRD+_densities_from_marginals")
def SEIRD_densities_from_marginals(
    data: xr.Dataset, *, num_samples: int, true_counts: xr.Dataset, cfg: dict, **kwargs
) -> xr.Dataset:

    return _densities_from_marginals(
        data,
        models.SEIRD.DataGeneration.generate_smooth_data,
        num_samples=num_samples,
        true_counts=true_counts,
        cfg=cfg,
        **kwargs,
    )


@is_operation("SEIRD+_densities_from_joint")
def SEIRD_densities_from_joint(
    data: xr.Dataset, prob: xr.Dataset, *, true_counts: xr.Dataset, cfg: dict, **kwargs
) -> xr.Dataset:

    return _densities_from_joint(
        data,
        prob,
        models.SEIRD.DataGeneration.generate_smooth_data,
        true_counts=true_counts,
        cfg=cfg,
        **kwargs,
    )


@is_operation("SEIRD+_densities_from_mode")
def SEIRD_densities_from_mode(
    data: xr.Dataset, prob: xr.Dataset, *, true_counts: xr.Dataset, cfg: dict, **kwargs
) -> xr.Dataset:

    return _densities_from_mode(
        data,
        prob,
        models.SEIRD.DataGeneration.generate_smooth_data,
        true_counts=true_counts,
        cfg=cfg,
        **kwargs,
    )


@is_operation("SEIRD+_residuals_and_accuracy")
def SEIRD_residuals_and_accuracy(
    data: xr.Dataset, prob: xr.Dataset, *, true_counts: xr.Dataset, cfg: dict, **kwargs
) -> xr.Dataset:

    return _accuracy_and_uncertainty(
        data,
        prob,
        models.SEIRD.DataGeneration.generate_smooth_data,
        true_counts=true_counts,
        cfg=cfg,
        **kwargs,
    )


@is_operation("SIR_residuals")
def SIR_residuals(data):

    residuals = _get_residuals(data)

    avgs = xr.DataArray(
        data=[
            np.repeat(
                np.abs(residuals)
                .sel({"type": "mode_residual"})
                .mean("time", skipna=True)
                .expand_dims({"time": [0]}, axis=0),
                len(data.coords["time"]),
                axis=0,
            ),
            np.repeat(
                np.abs(residuals)
                .sel({"type": "mean_residual"})
                .mean("time", skipna=True)
                .expand_dims({"time": [0]}, axis=0),
                len(data.coords["time"]),
                axis=0,
            ),
        ],
        dims=["type", "time", "kind"],
        coords=dict(
            type=["mode_average", "mean_average"],
            kind=data.coords["kind"],
            time=data.coords["time"],
        ),
    )
    l2_residual = np.sqrt((residuals**2).mean(["kind", "time"], skipna=True))

    log.remark(
        f"L1 residuals:      \n                              "
        f"mode prediction: {np.abs(residuals).sel({'type': 'mode_residual'}).mean(['time', 'kind'], skipna=True).item()}\n                              "
        f"mean prediction: {np.abs(residuals).sel({'type': 'mean_residual'}).mean(['time', 'kind'], skipna=True).item()}\n"
        f"                           L2 residuals:      \n                              "
        f"mode prediction: {l2_residual.sel({'type': 'mode_residual'}).item()}\n                              "
        f"mean prediction: {l2_residual.sel({'type': 'mean_residual'}).item()}"
    )

    return xr.concat([residuals, avgs], "type")
