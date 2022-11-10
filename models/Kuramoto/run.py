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

Kuramoto = import_module_from_path(mod_path=up(up(__file__)), mod_str="Kuramoto")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


# -----------------------------------------------------------------------------
# -- Model implementation -----------------------------------------------------
# -----------------------------------------------------------------------------


class Kuramoto_NN:
    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        training_data_group: h5.Group,
        pred_nw_group: h5.Group = None,
        neural_net: base.NeuralNet,
        loss_function: dict,
        ABM: Kuramoto.Kuramoto_ABM,
        true_network: torch.tensor,
        num_agents: int,
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
            num_agents: the number of agents in the model
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
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.num_agents = num_agents
        self.nw_size = num_agents**2
        self.true_network = true_network

        self._write_every = write_every
        self._write_predictions_every = write_predictions_every
        self._write_start = write_start
        self._num_steps = num_steps
        self._write_time = write_time

        # Current training loss, Frobenius error, and current predictions
        self.current_loss = torch.tensor(0.0)
        self.current_frob_error = torch.tensor(0.0)
        self.current_predictions = torch.zeros(self.nw_size)
        self.current_adjacency_matrix = torch.zeros(self.num_agents, self.num_agents)

        # Store the neural net training loss
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

        # Store the prediction error
        self._dset_frob_error = self._training_group.create_dataset(
            "frobenius_error",
            (0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression=3,
        )
        self._dset_frob_error.attrs["dim_names"] = ["time", "Frobenius error"]
        self._dset_frob_error.attrs["coords_mode__time"] = "start_and_step"
        self._dset_frob_error.attrs["coords__time"] = [write_start, write_every]

        # Store the neural net output, possibly less regularly than the loss
        self._dset_predictions = self._training_group.create_dataset(
            "predictions",
            (0, self.num_agents, self.num_agents),
            maxshape=(None, self.num_agents, self.num_agents),
            chunks=True,
            compression=3,
        )
        self._dset_predictions.attrs["dim_names"] = ["time", "i", "j"]
        self._dset_predictions.attrs["coords_mode__time"] = "start_and_step"
        self._dset_predictions.attrs["coords__time"] = [
            write_start,
            self._write_predictions_every,
        ]
        self._dset_predictions.attrs["coords_mode__i"] = "trivial"
        self._dset_predictions.attrs["coords_mode__j"] = "trivial"

        # Network vertices
        self._dset_vertices = self._pred_nw_group.create_dataset(
            "_vertices", (1, num_agents), chunks=True, compression=3, dtype=int
        )
        self._dset_vertices.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
        self._dset_vertices.attrs["coords_mode__vertex_idx"] = "trivial"
        self._dset_vertices[0, :] = np.arange(0, self.num_agents, 1)

        # Network edges
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

        # Predicted edge weights
        self._dset_edge_weights = pred_nw_group.create_dataset(
            "_edge_weights",
            (0, self.nw_size),
            maxshape=(None, self.num_agents**2),
            chunks=True,
            compression=3,
        )
        self._dset_edge_weights.attrs["dim_names"] = ["time", "edge_idx"]
        self._dset_edge_weights.attrs["coords_mode__time"] = "start_and_step"
        self._dset_edge_weights.attrs["coords__time"] = [
            write_start,
            write_predictions_every,
        ]
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

    def epoch(self, *, training_data, batch_size: int):

        start_time = time.time()

        for dset in training_data:

            batches = np.arange(0, len(dset), batch_size)
            if len(batches) == 1:
                batches = np.append(batches, len(dset) - 1)
            else:
                batches[-1] = len(dset) - 1

            for batch_no, batch_idx in enumerate(batches[:-1]):

                predicted_parameters = self.neural_net(torch.flatten(dset[batch_idx]))
                pred_adj_matrix = torch.reshape(
                    predicted_parameters, (self.num_agents, self.num_agents)
                )
                current_values = dset[batch_idx].clone()
                current_values.requires_grad_(True)

                loss = torch.tensor(0.0, requires_grad=True)

                for ele in range(batch_idx + 1, batches[batch_no + 1] + 1):

                    # Solve the ODE
                    current_values = self.ABM.run_single(
                        current_phases=current_values,
                        adjacency_matrix=pred_adj_matrix,
                        requires_grad=True,
                    )

                    # Calculate loss
                    loss = loss + self.loss_function(current_values, dset[ele]) / (
                        batches[batch_no + 1] - batch_idx
                    )

                # Penalise the trace (cannot be learned)
                loss = loss + torch.trace(pred_adj_matrix)

                loss = loss + torch.nn.MSELoss()(
                    pred_adj_matrix, torch.transpose(pred_adj_matrix, 0, 1)
                )

                loss.backward()
                self.neural_net.optimizer.step()
                self.neural_net.optimizer.zero_grad()
                self.current_loss = loss.clone().detach().numpy().item()
                self.current_frob_error = torch.nn.functional.mse_loss(
                    self.true_network,
                    torch.reshape(
                        self.current_predictions, (self.num_agents, self.num_agents)
                    ),
                )
                self.current_predictions = predicted_parameters.clone().detach()
                self.current_adjacency_matrix = pred_adj_matrix.clone().detach()
                self._time += 1
                self.write_data()
                self.write_predictions()

        if self._write_time:
            self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
            self.dset_time[-1, :] = time.time() - start_time

    def write_data(self):
        """Write the current loss and predicted network size into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start:

            if self._time % self._write_every == 0:
                self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
                self._dset_loss[-1, :] = self.current_loss

                self._dset_frob_error.resize(self._dset_frob_error.shape[0] + 1, axis=0)
                self._dset_frob_error[-1, :] = self.current_frob_error

                self._dset_nw_size.resize(self._dset_nw_size.shape[0] + 1, axis=0)
                self._dset_nw_size[-1, :] = torch.sum(
                    torch.ceil(self.current_predictions)
                )

    def write_predictions(self, *, write_final: bool = False):

        """Write the current predicted adjacency matrix into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """

        if self._write_predictions_every == -1 and not write_final:
            pass

        elif self._write_predictions_every != -1:
            if self._time % self._write_predictions_every != 0:
                pass

        else:
            log.info("    Writing prediction data ... ")
            self._dset_predictions.resize(self._dset_predictions.shape[0] + 1, axis=0)
            self._dset_predictions[-1, :] = self.current_adjacency_matrix

            # Write predicted network structure and edge weights, corresponding to the probability of that
            # edge existing. Write topological properties.
            # Create a graph from the current prediction
            G = nx.empty_graph(self.num_agents, create_using=nx.DiGraph)
            adj_matrix = self.current_adjacency_matrix

            curr_edges = torch.nonzero(adj_matrix).numpy()
            edge_weights = torch.flatten(
                self.current_predictions[torch.nonzero(self.current_predictions)]
            ).numpy()

            G.add_weighted_edges_from(
                np.column_stack([curr_edges, edge_weights]), weight="weight"
            )

            self._dset_edges.resize(self._dset_edges.shape[0] + 1, axis=0)
            self._dset_edges[-1, 0 : G.size(), :] = G.edges()

            self._dset_edge_weights.resize(self._dset_edge_weights.shape[0] + 1, axis=0)
            self._dset_edge_weights[-1, 0 : G.size()] = edge_weights

            self._dset_in_degree.resize(self._dset_in_degree.shape[0] + 1, axis=0)
            self._dset_in_degree[-1, :] = [deg[1] for deg in G.in_degree()]

            self._dset_in_degree_w.resize(self._dset_in_degree_w.shape[0] + 1, axis=0)
            self._dset_in_degree_w[-1, :] = [
                deg[1] for deg in G.in_degree(weight="weight")
            ]

            self._dset_out_degree.resize(self._dset_out_degree.shape[0] + 1, axis=0)
            self._dset_out_degree[-1, :] = [deg[1] for deg in G.out_degree()]

            self._dset_out_degree_w.resize(self._dset_out_degree_w.shape[0] + 1, axis=0)
            self._dset_out_degree_w[-1, :] = [
                deg[1] for deg in G.out_degree(weight="weight")
            ]

            self._dset_clustering.resize(self._dset_clustering.shape[0] + 1, axis=0)
            self._dset_clustering[-1, :] = [c for c in nx.clustering(G).values()]

            self._dset_clustering_w.resize(self._dset_clustering_w.shape[0] + 1, axis=0)
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
    model_name = cfg.get("root_model_name", "Kuramoto")
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
    training_data_group = h5file.create_group("training_data")
    neural_net_group = h5file.create_group("output_data")

    # Get the training data and the network, if synthetic data is used
    log.info("   Generating training data ...")
    training_data, network = Kuramoto.DataGeneration.get_data(
        model_cfg["Data"], h5file, training_data_group, seed=seed, device=device
    )

    # Get the eigen frequencies of the nodes
    eigen_frequencies = torch.stack(
        list(nx.get_node_attributes(network, "eigen_frequency").values())
    )

    # Generate the h5group for the predicted network, if it is to be learned
    num_agents = training_data.shape[2]
    predicted_nw_group = h5file.create_group("predicted_network")
    predicted_nw_group.attrs["content"] = "graph"
    predicted_nw_group.attrs["is_directed"] = True
    predicted_nw_group.attrs["allows_parallel"] = False

    output_size = num_agents**2

    log.info(
        f"   Initializing the neural net; input size: {num_agents}, output size: {output_size} ..."
    )
    net = base.NeuralNet(
        input_size=num_agents, output_size=output_size, **model_cfg["NeuralNet"]
    )

    # Get the true parameters
    true_parameters = model_cfg["Training"]["true_parameters"]

    # Initialise the ABM
    ABM = Kuramoto.Kuramoto_ABM(
        N=num_agents,
        dt=model_cfg["Data"]["synthetic_data"]["dt"],
        eigen_frequencies=eigen_frequencies,
    )

    # Calculate the frequency with which to write out the model predictions
    write_predictions_every = cfg.get("write_predictions_every", cfg["write_every"])
    num_epochs = cfg["num_epochs"]
    batch_size = model_cfg["Training"]["batch_size"]

    # Initialise the model
    model = Kuramoto_NN(
        model_name,
        rng=rng,
        training_data_group=neural_net_group,
        pred_nw_group=predicted_nw_group,
        num_agents=num_agents,
        neural_net=net,
        loss_function=model_cfg["Training"]["loss_function"],
        true_network=torch.from_numpy(nx.to_numpy_matrix(network)).float(),
        ABM=ABM,
        num_steps=training_data.shape[1],
        write_every=cfg["write_every"],
        write_predictions_every=write_predictions_every,
        write_start=cfg["write_start"],
        write_time=model_cfg.get("write_time", False),
    )

    log.info(f"   Initialized model '{model_name}'.")

    # Train the neural net
    log.info(f"   Now commencing training for {num_epochs} epochs ...")

    for i in range(num_epochs):
        model.epoch(training_data=training_data, batch_size=batch_size)

        log.progress(
            f"   Completed epoch {i + 1} / {num_epochs}; current loss: {model.current_loss}"
        )

    model.write_predictions(write_final=write_predictions_every == -1)

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")

    h5file.close()

    log.success("   All done.")
