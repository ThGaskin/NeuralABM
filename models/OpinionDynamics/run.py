#!/usr/bin/env python3
import sys
import time
from os.path import dirname as up

import coloredlogs
import h5py as h5
import networkx as nx
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

OpinionDynamics = import_module_from_path(
    mod_path=up(up(__file__)), mod_str="OpinionDynamics"
)
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


# -----------------------------------------------------------------------------
# -- Model implementation -----------------------------------------------------
# -----------------------------------------------------------------------------


class OpinionDynamics_NN:

    LOSS_FUNCTIONS = {
        "l1loss": torch.nn.L1Loss,
        "mseloss": torch.nn.MSELoss,
        "crossentropyloss": torch.nn.CrossEntropyLoss,
        "ctcloss": torch.nn.CTCLoss,
        "nllloss": torch.nn.NLLLoss,
        "poissonnllloss": torch.nn.PoissonNLLLoss,
        "gaussiannllloss": torch.nn.GaussianNLLLoss,
        "kldivloss": torch.nn.KLDivLoss,
        "bceloss": torch.nn.BCELoss,
        "bcewithlogitsloss": torch.nn.BCEWithLogitsLoss,
        "marginrankingloss": torch.nn.MarginRankingLoss,
        "hingeembeddingloss": torch.nn.HingeEmbeddingLoss,
        "multilabelmarginloss": torch.nn.MultiLabelMarginLoss,
        "huberloss": torch.nn.HuberLoss,
        "smoothl1loss": torch.nn.SmoothL1Loss,
        "softmarginloss": torch.nn.SoftMarginLoss,
        "multilabelsoftmarginloss": torch.nn.MultiLabelSoftMarginLoss,
        "cosineembeddingloss": torch.nn.CosineEmbeddingLoss,
        "multimarginloss": torch.nn.MultiMarginLoss,
        "tripletmarginloss": torch.nn.TripletMarginLoss,
        "tripletmarginwithdistanceloss": torch.nn.TripletMarginWithDistanceLoss
    }

    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        training_data_group: h5.Group,
        pred_nw_group: h5.Group = None,
        neural_net: base.NeuralNet,
        loss_function: dict,
        ABM: OpinionDynamics.OpinionDynamics_ABM,
        to_learn: list,
        num_agents: int,
        n_params: int,
        write_every: int = 1,
        write_predictions_every: int = 1,
        write_start: int = 1,
        num_steps: int = 3,
        write_time: bool = False,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            name (str): The name of this model instance
            rng (np.random.Generator): The shared RNG
            training_data_group (h5.Group): The output file group to write training data to
            pred_nw_group (h5.Group): (optional) The output file group to write the network edges to
            neural_net: The neural network
            ABM: The numerical solver
            to_learn: the list of parameter names to learn
            num_agents: the number of agents in the model
            n_params: the number of parameters to learn
            write_every: write every iteration
            write_predictions_every: write out predicted parameters every iteration
            write_start: iteration at which to start writing
            num_steps: number of iterations of the ABM
            write_time: whether to write out the training time into a dataset
        """
        self._name = name
        self._time = 0
        self._training_group = training_data_group
        self._pred_nw_group = pred_nw_group
        self._rng = rng

        self.ABM = ABM
        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = self.LOSS_FUNCTIONS[loss_function.get("name").lower()](loss_function.get("args", None),
                                                                                    **loss_function.get("kwargs", {}))

        self.n_params = n_params
        self.num_agents = num_agents
        self.nw_size = num_agents**2  # TODO: Generalise this to non-square matrices?

        self._write_every = write_every
        self._write_predictions_every = write_predictions_every
        self._write_start = write_start
        self._num_steps = num_steps
        self._write_time = write_time

        self.current_loss = torch.tensor(0.0)
        self.current_predictions = torch.zeros(self.n_params)

        # Store the neural net loss
        self._dset_loss = self._training_group.create_dataset(
            "loss",
            (0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["time", "training loss"]
        self._dset_loss.attrs["coords_mode__time"] = "start_and_step"
        self._dset_loss.attrs["coords__time"] = [write_start, write_every]

        # Neural net output
        self._dset_current_predictions = self._training_group.create_dataset(
            "current_predictions",
            (0, self.n_params),
            maxshape=(None, self.n_params),
            chunks=True,
            compression=3,
        )
        self._dset_current_predictions.attrs["dim_names"] = [
            "time",
            "current_predictions",
        ]
        self._dset_current_predictions.attrs["coords_mode__time"] = "start_and_step"
        self._dset_current_predictions.attrs["coords__time"] = [
            write_start,
            self._write_predictions_every,
        ]

        if "network" in to_learn:

            # Vertices
            self._dset_vertices = self._pred_nw_group.create_dataset(
                "_vertices", (1, num_agents), chunks=True, compression=3, dtype=int
            )
            self._dset_vertices.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
            self._dset_vertices.attrs["coords_mode__vertex_idx"] = "trivial"
            self._dset_vertices[0, :] = np.arange(0, self.num_agents, 1)

            # Edges
            self._dset_edges = self._pred_nw_group.create_dataset(
                "_edges",
                (0, self.nw_size, 2),
                maxshape=(None, self.nw_size, 2),
                chunks=True,
                compression=3,
            )
            self._dset_edges.attrs["dim_names"] = ["time", "edge_idx", "vertex_idx"]
            self._dset_edges.attrs["coords_mode__time"] = "start_and_step"
            self._dset_edges.attrs["coords__time"] = [write_start, write_predictions_every]
            self._dset_edges.attrs["coords_mode__edge_idx"] = "trivial"
            self._dset_edges.attrs["coords_mode__vertex_idx"] = "trivial"

            # Size of the network over time
            self._dset_nw_size = self._training_group.create_dataset(
                "network_size",
                (0, 1),
                maxshape=(None, 1),
                chunks=True,
                compression=3,
            )
            self._dset_nw_size.attrs["dim_names"] = ["time", "network size"]
            self._dset_nw_size.attrs["coords_mode__time"] = "start_and_step"
            self._dset_nw_size.attrs["coords__time"] = [write_start, write_every]

            # Edge weights (probability of that edge existing)
            self._dset_edge_weights = pred_nw_group.create_dataset(
                "_edge_weights",
                (0, self.nw_size),
                maxshape=(None, self.num_agents**2),
                chunks=True,
                compression=3,
            )
            self._dset_edge_weights.attrs["dim_names"] = ["time", "edge_idx"]
            self._dset_edge_weights.attrs["coords_mode__time"] = "start_and_step"
            self._dset_edge_weights.attrs["coords__time"] = [write_start, write_predictions_every]
            self._dset_edge_weights.attrs["coords_mode__edge_idx"] = "trivial"

            # In-degree
            self._dset_in_degree = predicted_nw_group.create_dataset(
                "_in_degree",
                (1, self.num_agents),
                maxshape=(None, self.num_agents),
                chunks=True,
                compression=3,
                dtype=int,
            )
            self._dset_in_degree.attrs["dim_names"] = ["time", "vertex_idx"]
            self._dset_in_degree.attrs["coords_mode__time"] = "start_and_step"
            self._dset_in_degree.attrs["coords__time"] = [
                write_start,
                self._write_predictions_every,
            ]

            # Weighted in-degree
            self._dset_in_degree_w = predicted_nw_group.create_dataset(
                "_in_degree_weighted",
                (1, self.num_agents),
                maxshape=(None, self.num_agents),
                chunks=True,
                compression=3,
                dtype=float,
            )
            self._dset_in_degree_w.attrs["dim_names"] = ["time", "vertex_idx"]
            self._dset_in_degree_w.attrs["coords_mode__time"] = "start_and_step"
            self._dset_in_degree_w.attrs["coords__time"] = [
                write_start,
                self._write_predictions_every,
            ]

            # Out-degree
            self._dset_out_degree = predicted_nw_group.create_dataset(
                "_out_degree",
                (1, self.num_agents),
                maxshape=(None, self.num_agents),
                chunks=True,
                compression=3,
                dtype=int,
            )
            self._dset_out_degree.attrs["dim_names"] = ["time", "vertex_idx"]
            self._dset_out_degree.attrs["coords_mode__time"] = "start_and_step"
            self._dset_out_degree.attrs["coords__time"] = [
                write_start,
                self._write_predictions_every,
            ]

            # Weighted out-degree
            self._dset_out_degree_w = predicted_nw_group.create_dataset(
                "_out_degree_weighted",
                (1, self.num_agents),
                maxshape=(None, self.num_agents),
                chunks=True,
                compression=3,
                dtype=float,
            )
            self._dset_out_degree_w.attrs["dim_names"] = ["time", "vertex_idx"]
            self._dset_out_degree_w.attrs["coords_mode__time"] = "start_and_step"
            self._dset_out_degree_w.attrs["coords__time"] = [
                write_start,
                self._write_predictions_every,
            ]

            # Clustering coefficients
            self._dset_clustering = predicted_nw_group.create_dataset(
                "_clustering",
                (1, self.num_agents),
                maxshape=(None, self.num_agents),
                chunks=True,
                compression=3,
                dtype=float,
            )
            self._dset_clustering.attrs["dim_names"] = ["time", "vertex_idx"]
            self._dset_clustering.attrs["coords_mode__time"] = "start_and_step"
            self._dset_clustering.attrs["coords__time"] = [
                write_start,
                self._write_predictions_every,
            ]

            # Weighted clustering coefficients
            self._dset_clustering_w = predicted_nw_group.create_dataset(
                "_clustering_weighted",
                (1, self.num_agents),
                maxshape=(None, self.num_agents),
                chunks=True,
                compression=3,
                dtype=float,
            )
            self._dset_clustering_w.attrs["dim_names"] = ["time", "vertex_idx"]
            self._dset_clustering_w.attrs["coords_mode__time"] = "start_and_step"
            self._dset_clustering_w.attrs["coords__time"] = [
                write_start,
                self._write_predictions_every,
            ]

        if write_time:
            self.dset_time = self._training_group.create_dataset(
                "computation_time",
                (0, 1),
                maxshape=(None, 1),
                chunks=True,
                compression=3,
            )
            self.dset_time.attrs["dim_names"] = ["epoch", "training_time"]
            self.dset_time.attrs["coords_mode__epoch"] = "trivial"
            self.dset_time.attrs["coords_mode__training_time"] = "trivial"

        self.to_learn = to_learn

    def epoch(self, *, training_data, batch_size: int):

        start_time = time.time()

        for s in range(self._num_steps - batch_size):

            predicted_parameters = self.neural_net(torch.flatten(training_data[s]))

            current_values = training_data[s].clone()
            current_values.requires_grad_(True)

            loss = torch.tensor(0.0, requires_grad=True)

            for ele in range(s + 1, s + batch_size + 1):

                # Solve the ODE
                current_values = self.ABM.run_single(
                    current_values=current_values,
                    input_data=predicted_parameters,
                    requires_grad=True,
                )

                # Calculate loss
                loss = (
                    loss
                    + self.loss_function(current_values, training_data[ele])
                    / batch_size
                )

            # Penalise the trace (cannot be learned)
            loss = (
                loss
                + torch.trace(
                    torch.reshape(
                        predicted_parameters, (self.num_agents, self.num_agents)
                    )
                )
            )

            loss.backward()
            self.neural_net.optimizer.step()
            self.neural_net.optimizer.zero_grad()
            self.current_loss = loss.clone().detach().numpy().item()
            self.current_predictions = predicted_parameters.clone().detach()
            self._time += 1
            self.write_data()

        if self._write_time:
            self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
            self.dset_time[-1, :] = time.time() - start_time

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start:

            if self._time % self._write_every == 0:
                self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
                self._dset_loss[-1, :] = self.current_loss

                if "network" in self.to_learn:
                    self._dset_nw_size.resize(self._dset_nw_size.shape[0] + 1, axis=0)
                    self._dset_nw_size[-1, :] = torch.sum(torch.ceil(self.current_predictions))

            if self._time % self._write_predictions_every == 0:

                log.info("    Writing prediction data ... ")
                self._dset_current_predictions.resize(
                    self._dset_current_predictions.shape[0] + 1, axis=0
                )
                self._dset_current_predictions[-1, :] = self.current_predictions

                # Write predicted network structure and edge weights, corresponding to the probability of that
                # edge existing. Write topological properties.
                if "network" in self.to_learn:

                    # Create a graph from the current prediction
                    G = nx.empty_graph(self.num_agents, create_using=nx.DiGraph)
                    adj_matrix = torch.reshape(
                        self.current_predictions, (self.num_agents, self.num_agents)
                    )
                    curr_edges = torch.nonzero(adj_matrix).numpy()
                    edge_weights = torch.flatten(
                        self.current_predictions[
                            torch.nonzero(self.current_predictions)
                        ]
                    ).numpy()

                    G.add_weighted_edges_from(
                        np.column_stack([curr_edges, edge_weights]), weight="weight"
                    )

                    self._dset_edges.resize(self._dset_edges.shape[0] + 1, axis=0)
                    self._dset_edges[-1, 0 : G.size(), :] = G.edges()

                    self._dset_edge_weights.resize(
                        self._dset_edge_weights.shape[0] + 1, axis=0
                    )
                    self._dset_edge_weights[-1, 0 : G.size()] = edge_weights

                    self._dset_in_degree.resize(
                        self._dset_in_degree.shape[0] + 1, axis=0
                    )
                    self._dset_in_degree[-1, :] = [deg[1] for deg in G.in_degree()]

                    self._dset_in_degree_w.resize(
                        self._dset_in_degree_w.shape[0] + 1, axis=0
                    )
                    self._dset_in_degree_w[-1, :] = [
                        deg[1] for deg in G.in_degree(weight="weight")
                    ]

                    self._dset_out_degree.resize(
                        self._dset_out_degree.shape[0] + 1, axis=0
                    )
                    self._dset_out_degree[-1, :] = [deg[1] for deg in G.out_degree()]

                    self._dset_out_degree_w.resize(
                        self._dset_out_degree_w.shape[0] + 1, axis=0
                    )
                    self._dset_out_degree_w[-1, :] = [
                        deg[1] for deg in G.out_degree(weight="weight")
                    ]

                    self._dset_clustering.resize(
                        self._dset_clustering.shape[0] + 1, axis=0
                    )
                    self._dset_clustering[-1, :] = [
                        c for c in nx.clustering(G).values()
                    ]

                    self._dset_clustering_w.resize(
                        self._dset_clustering_w.shape[0] + 1, axis=0
                    )
                    self._dset_clustering_w[-1, :] = [
                        c for c in nx.clustering(G, weight="weight").values()
                    ]


# ----------------------------------------------------------------------------------------------------------------------
# -- Performing the simulation run -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    with open(cfg_file_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "OpinionDynamics")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device to use
    training_device = model_cfg["Training"].get("device", None)
    if training_device is not None:
        device = training_device
    else:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Set number of threads to use
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    log.note("   Creating global RNG ...")
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    op_data_group = h5file.create_group("opinion_data")
    training_data_group = h5file.create_group("training_data")

    # Get the training data and the network, if synthetic data is used
    log.info("   Generating training data ...")
    training_data, network = OpinionDynamics.DataGeneration.get_data(
        model_cfg["Data"], h5file, op_data_group, seed=seed, device=device
    )

    # Generate the h5group for the predicted network, if it is to be learned
    num_agents = training_data.shape[1]
    to_learn = model_cfg["Training"]["to_learn"]
    predicted_nw_group = (
        h5file.create_group("predicted_network") if "network" in to_learn else None
    )

    if predicted_nw_group is not None:
        predicted_nw_group.attrs["content"] = "graph"
        predicted_nw_group.attrs["is_directed"] = True
        predicted_nw_group.attrs["allows_parallel"] = False

    output_size = (
        len(to_learn)
        if "network" not in to_learn
        else len(to_learn) + num_agents**2 - 1
    )

    log.info(
        f"   Initializing the neural net; input size: {num_agents}, output size: {output_size} ..."
    )
    net = base.NeuralNet(
        input_size=num_agents, output_size=output_size, **model_cfg["NeuralNet"]
    )

    # Get the true parameters
    true_parameters = model_cfg["Training"]["true_parameters"]

    # Initialise the ABM
    ABM = OpinionDynamics.OpinionDynamics_ABM(
        N=num_agents,
        network=None if "network" in to_learn else network,
        init_values=training_data[0],
        **true_parameters,
    )

    # Calculate the frequency with which to write out the model predictions
    write_predictions_every = cfg.get("write_predictions_every", None)
    num_epochs = cfg["num_epochs"]
    batch_size = model_cfg["Training"]["batch_size"]
    if write_predictions_every is None:
        write_predictions_every = cfg["write_every"]
    elif write_predictions_every == -1:
        write_predictions_every = (len(training_data) - batch_size) * num_epochs

    # Initialise the model
    model = OpinionDynamics_NN(
        model_name,
        rng=rng,
        training_data_group=training_data_group,
        pred_nw_group=predicted_nw_group,
        num_agents=num_agents,
        n_params=output_size,
        neural_net=net,
        loss_function=model_cfg["Training"]["loss_function"],
        ABM=ABM,
        to_learn=model_cfg["Training"]["to_learn"],
        num_steps=len(training_data),
        write_every=cfg["write_every"],
        write_predictions_every=write_predictions_every,
        write_start=cfg["write_start"],
        write_time=model_cfg.get("write_time", False),
    )

    log.info(f"   Initialized model '{model_name}'.")

    # Train the neural net
    log.info(f"   Now commencing training for {num_epochs} epochs ...")

    for i in range(num_epochs):
        model.epoch(
            training_data=training_data, batch_size=batch_size
        )
        log.progress(f"   Completed epoch {i + 1} / {num_epochs}; current loss: {model.current_loss}")

    log.progress("   Generating predicted dataset ...")

    # Generate a predicted time series using the neural net prediction:
    num_steps = training_data.shape[0] - 1
    pred_opinions = op_data_group.create_dataset(
        "predicted_opinions",
        (num_steps + 1, num_agents),
        maxshape=(num_steps + 1, num_agents),
        chunks=True,
        compression=3,
    )
    pred_opinions.attrs["dim_names"] = ["time", "vertex_idx"]
    pred_opinions.attrs["coords_mode__time"] = "start_and_step"
    pred_opinions.attrs["coords__time"] = [0, 1]
    pred_opinions.attrs["coords_mode__vertex_idx"] = "trivial"

    current_values = ABM.initial_opinions
    pred_opinions[0, :] = torch.flatten(current_values)

    for _ in range(num_steps):

        current_values = ABM.run_single(
            current_values=current_values,
            input_data=model.current_predictions,
            requires_grad=False,
        )
        pred_opinions[_ + 1, :] = torch.flatten(current_values)

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")

    h5file.close()

    log.success("   All done.")
