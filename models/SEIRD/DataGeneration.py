import logging

import h5py as h5
import numpy as np
import torch

from .ABM import SEIRD_ABM

log = logging.getLogger(__name__)


# --- Data generation functions ------------------------------------------------------------------------------------
def generate_data_from_ABM(
    *,
    cfg: dict,
    positions=None,
    kinds=None,
    counts=None,
    write_init_state: bool = True,
    burn_in: int = 0,
):
    """
    Runs the ABM for n iterations and writes out the data, if datasets are passed.

    :param cfg: the data generation configuration settings
    :param parameters: (optional) the parameters to use to run the model. Defaults to the ABM defaults
    :param positions: (optional) the dataset to write the agent positions to
    :kinds: (optional) the dataset to write the ABM kinds to
    :counts: (optional) the dataset to write the ABM counts to
    """

    log.info("   Initialising the ABM ... ")

    num_steps: int = cfg["synthetic_data"]["num_steps"]

    ABM = SEIRD_ABM(**cfg["synthetic_data"])
    data = (
        torch.empty((num_steps + 1, 12, 1), dtype=torch.float)
        if write_init_state
        else torch.empty((num_steps, 12, 1))
    )

    if write_init_state:
        data[0, :] = ABM.current_counts.float() / ABM.N

    log.info("   Generating synthetic data ... ")
    for time in range(num_steps + burn_in):

        # Run the ABM for a single step
        ABM.run_single(time=time)

        # Get the densities
        densities = ABM.current_counts.float() / ABM.N

        if time < burn_in:
            continue

        # Write out the new positions
        if positions:
            positions.resize(positions.shape[0] + 1, axis=0)
            positions[-1, :, :] = ABM.current_positions

        # Write out the new kinds
        if kinds:
            kinds.resize(kinds.shape[0] + 1, axis=0)
            kinds[-1, :] = ABM.current_kinds

        # Write out the new counts
        if counts:
            counts.resize(counts.shape[0] + 1, axis=0)
            counts[-1, :] = densities

        # Append the new counts to training dataset
        data[time] = densities

        log.debug(f"   Completed run {time} of {num_steps} ... ")

    return data


def generate_smooth_data(
    *,
    cfg: dict = None,
    num_steps: int = None,
    dt: float = None,
    k_q: float = 10.25,
    parameters=None,
    init_state: torch.tensor,
    counts=None,
    write_init_state: bool = True,
    requires_grad: bool = False,
    burn_in: int = 0,
):
    """
    Generates a dataset of SIR-counts by iteratively solving the system of differential equations.
    """

    num_steps: int = cfg["num_steps"] if num_steps is None else num_steps
    dt: float = cfg["dt"] if dt is None else dt
    k_q: float = cfg["k_q"] if k_q is None else k_q

    data = (
        torch.empty((num_steps, 12, 1), dtype=torch.float)
        if not write_init_state
        else torch.empty((num_steps + 1, 12, 1), dtype=torch.float)
    )
    k_S = (
        torch.tensor(cfg["k_S"], dtype=torch.float)
        if parameters is None
        else parameters[0]
    )
    k_E = (
        torch.tensor(cfg["k_E"], dtype=torch.float)
        if parameters is None
        else parameters[1]
    )
    k_I = (
        torch.tensor(cfg["k_I"], dtype=torch.float)
        if parameters is None
        else parameters[2]
    )
    k_R = (
        torch.tensor(cfg["k_R"], dtype=torch.float)
        if parameters is None
        else parameters[3]
    )
    k_SY = (
        torch.tensor(cfg["k_SY"], dtype=torch.float)
        if parameters is None
        else parameters[4]
    )
    k_H = (
        torch.tensor(cfg["k_H"], dtype=torch.float)
        if parameters is None
        else parameters[5]
    )
    k_C = (
        torch.tensor(cfg["k_C"], dtype=torch.float)
        if parameters is None
        else parameters[6]
    )
    k_D = (
        torch.tensor(cfg["k_D"], dtype=torch.float)
        if parameters is None
        else parameters[7]
    )
    k_CT = (
        torch.tensor(cfg["k_CT"], dtype=torch.float)
        if parameters is None
        else parameters[8]
    )

    # Write out the initial state if required
    if write_init_state:
        data[0] = init_state
        if counts:
            counts.resize(counts.shape[0] + 1, axis=0)
            counts[-1, :] = init_state

    current_densities = init_state.clone()
    current_densities.requires_grad = requires_grad

    for t in range(num_steps + burn_in):

        # Get the time-dependent parameters
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
        k_Q_t = k_q * k_CT_t * current_densities[-1]

        # Solve the ODE
        current_densities = torch.clip(
            current_densities
            + torch.stack(
                [
                    (-k_E_t * current_densities[2] - k_Q_t) * current_densities[0]
                    + k_S_t * current_densities[8],
                    k_E_t * current_densities[0] * current_densities[2]
                    - (k_I_t + k_Q_t) * current_densities[1],
                    k_I_t * current_densities[1]
                    - (k_R_t + k_SY_t + k_Q_t) * current_densities[2],
                    k_R_t
                    * (
                        current_densities[2]
                        + current_densities[4]
                        + current_densities[5]
                        + current_densities[6]
                        + current_densities[10]
                    ),
                    k_SY_t * (current_densities[2] + current_densities[10])
                    - (k_R_t + k_H_t) * current_densities[4],
                    k_H_t * current_densities[4]
                    - (k_R_t + k_C_t) * current_densities[5],
                    k_C_t * current_densities[5]
                    - (k_R_t + k_D_t) * current_densities[6],
                    k_D_t * current_densities[6],
                    -k_S_t * current_densities[8] + k_Q_t * current_densities[0],
                    -k_I_t * current_densities[9] + k_Q_t * current_densities[1],
                    k_I_t * current_densities[9]
                    + k_Q_t * current_densities[2]
                    - (k_SY_t + k_R_t) * current_densities[10],
                    k_SY_t * current_densities[2]
                    - k_q * torch.sum(current_densities[0:3]) * current_densities[-1],
                ]
            )
            * dt,
            0,
            1,
        )
        if t < burn_in:
            continue
        if write_init_state:
            data[t + 1 - burn_in] = current_densities
        else:
            data[t - burn_in] = current_densities

        if counts:
            counts.resize(counts.shape[0] + 1, axis=0)
            counts[-1, :] = current_densities

    return data


def get_SIR_data(*, data_cfg: dict, h5group: h5.Group, write_init_state: bool = False):
    """Returns the training data for the SIR model. If a directory is passed, the
    data is loaded from that directory. Otherwise, synthetic training data is generated, either from an ABM,
    or by iteratively solving the temporal ODE system.
    """
    if "load_from_dir" in data_cfg.keys():

        with h5.File(data_cfg["load_from_dir"], "r") as f:

            data = np.array(f["SEIRD+"]["true_counts"])

            dset_true_counts = h5group.create_dataset(
                "true_counts",
                np.shape(data),
                maxshape=np.shape(data),
                chunks=True,
                compression=3,
                dtype=float,
            )

            dset_true_counts.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
            dset_true_counts.attrs["coords_mode__time"] = "trivial"
            dset_true_counts.attrs["coords_mode__kind"] = "values"
            dset_true_counts.attrs["coords__kind"] = [
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
            ]
            dset_true_counts.attrs["coords_mode__dim_name__0"] = "trivial"

            dset_true_counts[:, :, :] = data

            return torch.from_numpy(data).float()

    elif "synthetic_data" in data_cfg.keys():

        time_dependent_params: dict = data_cfg.get("time_dependent_params", {})
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

        # True counts
        dset_true_counts = h5group.create_dataset(
            "true_counts",
            (0, 12, 1),
            maxshape=(None, 12, 1),
            chunks=True,
            compression=3,
            dtype=float,
        )

        dset_true_counts.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
        dset_true_counts.attrs["coords_mode__time"] = "trivial"
        dset_true_counts.attrs["coords_mode__kind"] = "values"
        dset_true_counts.attrs["coords__kind"] = [
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
        ]
        dset_true_counts.attrs["coords_mode__dim_name__0"] = "trivial"

        # --- Generate the data ----------------------------------------------------------------------------------------
        type = data_cfg["synthetic_data"]["type"]

        if type == "smooth":
            N = data_cfg["synthetic_data"]["N"]
            init_state = (
                torch.tensor(
                    [
                        [N - 1],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype=torch.float,
                )
                / N
            )
            training_data = generate_smooth_data(
                cfg=data_cfg["synthetic_data"],
                init_state=init_state,
                counts=dset_true_counts,
                write_init_state=write_init_state,
                burn_in=burn_in,
            )

        elif type == "from_ABM":

            N = data_cfg["synthetic_data"]["N"]

            # Initialise agent position dataset
            dset_position = h5group.create_dataset(
                "position",
                (0, N, 2),
                maxshape=(None, N, 2),
                chunks=True,
                compression=3,
            )
            dset_position.attrs["dim_names"] = ["time", "agent_id", "coords"]
            dset_position.attrs["coords_mode__time"] = "trivial"
            dset_position.attrs["coords_mode__agent_id"] = "trivial"
            dset_position.attrs["coords_mode__coords"] = "values"
            dset_position.attrs["coords__coords"] = ["x", "y"]

            # Initialise agent kind dataset
            dset_kinds = h5group.create_dataset(
                "kinds",
                (0, N),
                maxshape=(None, N),
                chunks=True,
                compression=3,
            )
            dset_kinds.attrs["dim_names"] = ["time", "agent_id"]
            dset_kinds.attrs["coords_mode__time"] = "trivial"
            dset_kinds.attrs["coords_mode__agent_id"] = "trivial"

            training_data = generate_data_from_ABM(
                cfg=data_cfg,
                positions=dset_position,
                kinds=dset_kinds,
                counts=dset_true_counts,
                write_init_state=write_init_state,
                burn_in=burn_in,
            )
        else:
            raise ValueError(
                f"Unrecognised arugment {type}! 'Type' must be one of 'smooth' or 'from_ABM'!"
            )

        return training_data

    else:
        raise ValueError(
            f"You must supply one of 'load_from_dir' or 'synthetic data' keys!"
        )
