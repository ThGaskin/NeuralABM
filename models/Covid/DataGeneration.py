import logging

import h5py as h5
import numpy as np
import torch

from .kinds import Compartments

log = logging.getLogger(__name__)


def generate_smooth_data(
    cfg, *, parameters=None, init_state: torch.tensor = None
) -> torch.Tensor:
    """Generates a dataset of counts for each compartment by iteratively solving the system of differential equations.

    :param cfg: configuration file, containing parameter values (possibly as a ``Sequence``, if time-dependent),
        number of steps, burn-in period, etc.
    :param parameters: (optional) parameters used to override cfg settings
    :param init_state: (optional) initial state to use; defaults to a generic density if ``None``
    :return: ``torch.Tensor`` training dataset, with the burn-in period discarded
    """

    # Get config settings
    num_steps: int = cfg["num_steps"]
    burn_in: int = cfg.get("burn_in", 0)
    dt: float = cfg["dt"]
    k_q: float = cfg.get("k_q", 10.25)

    # Use a generic initial state if None passed
    if init_state is None:
        init_state = torch.zeros(12, 1, dtype=torch.float)
        init_state[
            Compartments.susceptible.value
        ] = 0.9933  # High number of susceptible agents
        init_state[Compartments.infected.value] = (
            1.0 - init_state[Compartments.susceptible.value]
        )  # Some infected agents

    # Empty dataset for counts: the initial state is always written
    data = torch.empty((num_steps + burn_in, 12, 1), dtype=torch.float)
    data[0, :] = init_state

    # Get the model parameters; these can be overridden with the ``parameters`` argument
    k_S = (
        torch.tensor(cfg["k_S"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.susceptible.value]
    )
    k_E = (
        torch.tensor(cfg["k_E"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.exposed.value]
    )
    k_I = (
        torch.tensor(cfg["k_I"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.infected.value]
    )
    k_R = (
        torch.tensor(cfg["k_R"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.recovered.value]
    )
    k_SY = (
        torch.tensor(cfg["k_SY"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.symptomatic.value]
    )
    k_H = (
        torch.tensor(cfg["k_H"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.hospitalized.value]
    )
    k_C = (
        torch.tensor(cfg["k_C"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.critical.value]
    )
    k_D = (
        torch.tensor(cfg["k_D"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.deceased.value]
    )
    k_CT = (
        torch.tensor(cfg["k_CT"], dtype=torch.float)
        if parameters is None
        else parameters[Compartments.contact_traced.value]
    )

    # Solve the ODE
    for t in range(1, num_steps + burn_in):
        # Get the time-dependent parameters, if given
        k_S_t = k_S[t] if k_S.dim() > 0 else k_S
        k_E_t = k_E[t] if k_E.dim() > 0 else k_E
        k_I_t = k_I[t] if k_I.dim() > 0 else k_I
        k_R_t = k_R[t] if k_R.dim() > 0 else k_R
        k_SY_t = k_SY[t] if k_SY.dim() > 0 else k_SY
        k_H_t = k_H[t] if k_H.dim() > 0 else k_H
        k_C_t = k_C[t] if k_C.dim() > 0 else k_C
        k_D_t = k_D[t] if k_D.dim() > 0 else k_D
        k_CT_t = k_CT[t] if k_CT.dim() > 0 else k_CT

        # Calculate k_Q
        k_Q_t = k_q * k_CT_t * data[t - 1][Compartments.contact_traced.value]

        dy = torch.stack(
            [
                (-k_E_t * data[t - 1][Compartments.infected.value] - k_Q_t)
                * data[t - 1][Compartments.susceptible.value]
                + k_S_t * data[t - 1][Compartments.quarantine_S.value],
                k_E_t
                * data[t - 1][Compartments.susceptible.value]
                * data[t - 1][Compartments.infected.value]
                - (k_I_t + k_Q_t) * data[t - 1][Compartments.exposed.value],
                k_I_t * data[t - 1][Compartments.exposed.value]
                - (k_R_t + k_SY_t + k_Q_t) * data[t - 1][Compartments.infected.value],
                k_R_t
                * (
                    data[t - 1][Compartments.infected.value]
                    + data[t - 1][Compartments.symptomatic.value]
                    + data[t - 1][Compartments.hospitalized.value]
                    + data[t - 1][Compartments.critical.value]
                    + data[t - 1][Compartments.quarantine_I.value]
                ),
                k_SY_t
                * (
                    data[t - 1][Compartments.infected.value]
                    + data[t - 1][Compartments.quarantine_I.value]
                )
                - (k_R_t + k_H_t) * data[t - 1][Compartments.symptomatic.value],
                k_H_t * data[t - 1][Compartments.symptomatic.value]
                - (k_R_t + k_C_t) * data[t - 1][Compartments.hospitalized.value],
                k_C_t * data[t - 1][Compartments.hospitalized.value]
                - (k_R_t + k_D_t) * data[t - 1][Compartments.critical.value],
                k_D_t * data[t - 1][Compartments.critical.value],
                -k_S_t * data[t - 1][Compartments.quarantine_S.value]
                + k_Q_t * data[t - 1][Compartments.susceptible.value],
                -k_I_t * data[t - 1][Compartments.quarantine_E.value]
                + k_Q_t * data[t - 1][Compartments.exposed.value],
                k_I_t * data[t - 1][Compartments.quarantine_E.value]
                + k_Q_t * data[t - 1][Compartments.infected.value]
                - (k_SY_t + k_R_t) * data[t - 1][Compartments.quarantine_I.value],
                k_SY_t * data[t - 1][Compartments.infected.value]
                - k_q
                * torch.sum(data[t - 1][0:3])
                * data[t - 1][Compartments.contact_traced.value],
            ]
        )

        # Solve the ODE (simple forward Euler)
        data[t, :] = torch.clip(data[t - 1, :] + dy * dt, 0, 1)

    # Return the data, discarding the burn-in, if specified
    return data[burn_in:]


def get_data(data_cfg: dict, h5group: h5.Group) -> torch.Tensor:
    """Returns the training data for the Covid model. If a directory is passed, the data is loaded from that directory.
    Otherwise, synthetic training data is generated by iteratively solving the ODE system.

    :param data_cfg: configuration file
    :param h5group: hdf5.group to which to write the data
    :return: ``torch.Tensor`` training data
    """

    # Load training data from file
    if "load_from_dir" in data_cfg.keys():
        log.info("   Loading training data ...")
        # Load training data from hdf5 file
        with h5.File(data_cfg["load_from_dir"], "r") as f:
            training_data = torch.from_numpy(
                np.array(f["Covid"]["true_counts"])
            ).float()

    # Generate synthetic data
    elif "synthetic_data" in data_cfg.keys():
        log.info("   Generating training data ...")
        # Get the time dependent parameters: names and intervals
        time_dependent_params: dict = data_cfg.get("time_dependent_parameters", {})
        num_steps: int = data_cfg["synthetic_data"]["num_steps"]
        burn_in: int = data_cfg["synthetic_data"].get("burn_in", 0)

        # Replace any time-dependent parameters with a sequence
        for key in time_dependent_params.keys():
            p = np.zeros(num_steps + burn_in)
            i = 0
            for j, interval in enumerate(time_dependent_params[key]):
                _, upper = interval
                if not upper:
                    upper = num_steps
                while i < upper + burn_in:
                    p[i] = data_cfg["synthetic_data"][key][j]
                    i += 1
            data_cfg["synthetic_data"][key] = p

        # Generate training data by integrating the model equations
        training_data = generate_smooth_data(data_cfg["synthetic_data"])

    else:
        raise ValueError(
            f"You must supply one of 'load_from_dir' or 'synthetic data' keys!"
        )

    # Save training data to hdf5 dataset and return
    dset_true_counts = h5group.create_dataset(
        "true_counts",
        training_data.shape,
        maxshape=training_data.shape,
        chunks=True,
        compression=3,
        dtype=float,
    )

    dset_true_counts.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
    dset_true_counts.attrs["coords_mode__time"] = "trivial"
    dset_true_counts.attrs["coords_mode__kind"] = "values"
    dset_true_counts.attrs["coords__kind"] = [k.name for k in Compartments]
    dset_true_counts.attrs["coords_mode__dim_name__0"] = "trivial"

    dset_true_counts[:, :, :] = training_data

    return training_data
