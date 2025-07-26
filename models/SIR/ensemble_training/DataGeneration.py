import logging

import h5py as h5
import numpy as np
import torch

from .ABM import SIR_ABM

log = logging.getLogger(__name__)


# --- Data generation functions ------------------------------------------------------------------------------------
def generate_data_from_ABM(
    *,
    cfg: dict,
    parameters=None,
    positions=None,
    kinds=None,
    counts=None,
    write_init_state: bool = True,
    **__,
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

    ABM = SIR_ABM(**cfg)
    num_steps: int = cfg["num_steps"]
    data = (
        torch.empty((num_steps + 1, 3, 1), dtype=torch.float)
        if write_init_state
        else torch.empty((num_steps, 3, 1))
    )

    if write_init_state:
        data[0, :, :] = ABM.current_counts.float() / ABM.N

    parameters = (
        torch.tensor([ABM.p_infect, ABM.t_infectious])
        if parameters is None
        else parameters
    )

    log.info("   Generating synthetic data ... ")
    for _ in range(num_steps):
        # Run the ABM for a single step
        ABM.run_single(parameters=parameters)

        # Get the densities
        densities = ABM.current_counts.float() / ABM.N

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
        data[_] = densities

        log.debug(f"   Completed run {_} of {num_steps} ... ")

    return data


def generate_smooth_data(
    *,
    cfg: dict = None,
    num_steps: int = None,
    parameters=None,
    init_state: torch.tensor,
    counts=None,
    write_init_state: bool = True,
    requires_grad: bool = False,
    **__,
):
    """
    Generates a dataset of SIR-counts by iteratively solving the system of differential equations.
    """

    num_steps: int = cfg["num_steps"] if num_steps is None else num_steps
    data = (
        torch.empty((num_steps, 3, 1), dtype=torch.float)
        if not write_init_state
        else torch.empty((num_steps + 1, 3, 1), dtype=torch.float)
    )

    parameters = (
        torch.tensor(
            [cfg["p_infect"], cfg["t_infectious"], cfg["sigma"]], dtype=torch.float
        )
        if parameters is None
        else parameters
    )

    # Write out the initial state if required
    if write_init_state:
        data[0] = init_state
        if counts:
            counts.resize(counts.shape[0] + 1, axis=0)
            counts[-1, :] = init_state

    current_state = init_state.clone()
    current_state.requires_grad = requires_grad

    for _ in range(num_steps):
        # Generate the transformation matrix
        # Patients only start recovering after a certain time
        w = torch.normal(torch.tensor(0.0), torch.tensor(1.0))
        tau = 1 / parameters[1] * torch.sigmoid(1000 * (_ / parameters[1] - 1))
        matrix = torch.vstack(
            [
                torch.tensor([-parameters[0], -parameters[2] * w]),
                torch.tensor([parameters[0], -tau + parameters[2] * w]),
                torch.tensor([0, tau]),
            ]
        )
        current_state = torch.clip(
            current_state
            + torch.matmul(
                matrix,
                torch.vstack([current_state[0] * current_state[1], current_state[1]]),
            ),
            0.0,
            1.0,
        )

        if write_init_state:
            data[_ + 1] = current_state
        else:
            data[_] = current_state

        if counts:
            counts.resize(counts.shape[0] + 1, axis=0)
            counts[-1, :] = current_state

    return data


def get_SIR_data(*, data_cfg: dict, h5group: h5.Group, write_init_state: bool = True):
    """Returns the training data for the SIR model. If a directory is passed, the
    data is loaded from that directory. Otherwise, synthetic training data is generated, either from an ABM,
    or by iteratively solving the temporal ODE system.
    """
    if "load_from_dir" in data_cfg.keys():
        with h5.File(data_cfg["load_from_dir"], "r") as f:
            data = np.array(f["SIR"]["true_counts"])

            dset_true_counts = h5group.create_dataset(
                "true_counts",
                (len(data), 3, 1),
                maxshape=(None, 3, 1),
                chunks=True,
                compression=3,
                dtype=float,
            )

            dset_true_counts.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
            dset_true_counts.attrs["coords_mode__time"] = "trivial"
            dset_true_counts.attrs["coords_mode__kind"] = "values"
            dset_true_counts.attrs["coords__kind"] = [
                "susceptible",
                "infected",
                "recovered",
            ]
            dset_true_counts.attrs["coords_mode__dim_name__0"] = "trivial"

            dset_true_counts[:, :, :] = data

            return torch.from_numpy(data).float()

    elif "synthetic_data" in data_cfg.keys():
        # True counts
        dset_true_counts = h5group.create_dataset(
            "true_counts",
            (0, 3, 1),
            maxshape=(None, 3, 1),
            chunks=True,
            compression=3,
            dtype=float,
        )

        dset_true_counts.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
        dset_true_counts.attrs["coords_mode__time"] = "trivial"
        dset_true_counts.attrs["coords_mode__kind"] = "values"
        dset_true_counts.attrs["coords__kind"] = [
            "susceptible",
            "infected",
            "recovered",
        ]
        dset_true_counts.attrs["coords_mode__dim_name__0"] = "trivial"

        # --- Generate the data ----------------------------------------------------------------------------------------
        type = data_cfg["synthetic_data"]["type"]

        if type == "smooth":
            N = data_cfg["synthetic_data"]["N"]
            init_state = torch.tensor([[N - 1], [1], [0]], dtype=torch.float) / N
            training_data = generate_smooth_data(
                cfg=data_cfg["synthetic_data"],
                init_state=init_state,
                counts=dset_true_counts,
                write_init_state=write_init_state,
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
                cfg=data_cfg["synthetic_data"],
                positions=dset_position,
                kinds=dset_kinds,
                counts=dset_true_counts,
                write_init_state=write_init_state,
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
