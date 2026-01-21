import sys
from os.path import dirname as up

import h5py as h5
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import torch
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

HW = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="HarrisWilsonNW")

""" Generate a synthetic time series based on the London GLA data and using the GoogleMaps transport time metric """

np.random.seed(0)
torch.random.manual_seed(0)

with open("models/HarrisWilsonNW/cfgs/London_dataset/run.yml") as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.Loader)

data_cfg = (
    cfg.get("parameter_space").get("HarrisWilsonNW").get("Data").get("synthetic_data")
)
network = torch.from_numpy(
    pd.read_csv(
        "data/HarrisWilson/London_data/exp_times.csv", header=0, index_col=0
    ).to_numpy()
)
norms = torch.sum(network, dim=1, keepdim=True)
network /= torch.where(norms != 0, norms, 1)

N_origin = network.shape[0]
N_destination = network.shape[1]

num_steps: int = data_cfg.get("num_steps")
origin_size_std = data_cfg.get("origin_size_std")
alpha = data_cfg.get("alpha")
beta = data_cfg.get("beta")
epsilon = data_cfg.get("epsilon")
sigma = data_cfg.get("sigma")
dt = data_cfg.get("dt")
kappa = data_cfg.get("kappa")

num_training_steps = (
    cfg.get("parameter_space")
    .get("HarrisWilsonNW")
    .get("Data")
    .get("num_training_steps")
)
training_set_size = (
    cfg.get("parameter_space")
    .get("HarrisWilsonNW")
    .get("Data")
    .get("training_set_size")
)

h5file = h5.File(
    "/users/thomasgaskin/NeuralABM/data/HarrisWilsonNW/London_data.h5", mode="w"
)
h5group = h5file.create_group("training_data")

origin_sizes = torch.empty((num_steps, N_origin, 1))
origin_sizes[0, :, :] = torch.from_numpy(
    pd.read_csv(
        "data/HarrisWilson/London_data/origin_sizes.csv", header=0, index_col=0
    ).to_numpy()
)

destination_sizes = torch.empty((num_steps, N_destination, 1))

destination_sizes[0, :, 0] = torch.from_numpy(
    pd.read_csv(
        "data/HarrisWilson/London_data/dest_sizes.csv", header=0, index_col=0
    ).to_numpy()
)


# Initialise the ABM
ABM = HW.HarrisWilsonABM(
    N=N_origin,
    M=N_destination,
    dt=dt,
    device="cpu",
    alpha=alpha,
    beta=beta,
    kappa=kappa,
    sigma=sigma,
    epsilon=epsilon,
)

for __ in range(1, num_steps):
    origin_sizes[__, :, :] = origin_sizes[__ - 1, :, :] + torch.normal(
        0, origin_size_std, size=(N_origin, 1)
    )

origin_sizes = torch.abs(origin_sizes)

# Run the ABM for n iterations, generating the entire time series
destination_sizes[:, :, :] = ABM.run(
    init_data=destination_sizes[0, :, :],
    adjacency_matrix=network,
    n_iterations=num_steps - 1,
    origin_sizes=origin_sizes[:],
    generate_time_series=True,
)

# Subsample the time series by splitting it into chunks
destination_sizes = torch.stack(
    torch.split(
        destination_sizes,
        training_set_size * [int(len(destination_sizes) / training_set_size)],
    )
)[:, :num_training_steps]
origin_sizes = torch.stack(
    torch.split(
        origin_sizes, training_set_size * [int(len(origin_sizes) / training_set_size)]
    )
)[:, :num_training_steps]

# Set up dataset for complete synthetic time series
dset_dest_zones = h5group.create_dataset(
    "destination_sizes",
    destination_sizes.shape,
    maxshape=destination_sizes.shape,
    chunks=True,
    compression=3,
)
dset_dest_zones.attrs["dim_names"] = [
    "training_set",
    "time",
    "zone_id",
    "dim_name__0",
]
dset_dest_zones.attrs["coords_mode__training_set"] = "trivial"
dset_dest_zones.attrs["coords_mode__time"] = "trivial"
dset_dest_zones.attrs["coords_mode__zone_id"] = "values"
dset_dest_zones.attrs["coords__zone_id"] = np.arange(
    N_origin, N_origin + N_destination, 1
)

# Write the time series data
dset_dest_zones[:,] = destination_sizes

# Set up chunked dataset to store the state data in
# Origin zone sizes
dset_origin_sizes = h5group.create_dataset(
    "origin_sizes",
    origin_sizes.shape,
    maxshape=origin_sizes.shape,
    chunks=True,
    compression=3,
)
dset_origin_sizes.attrs["dim_names"] = [
    "training_set",
    "time",
    "zone_id",
    "dim_name__0",
]
dset_origin_sizes.attrs["coords_mode__training_set"] = "trivial"
dset_origin_sizes.attrs["coords_mode__time"] = "trivial"
dset_origin_sizes.attrs["coords_mode__zone_id"] = "values"
dset_origin_sizes.attrs["coords__zone_id"] = np.arange(0, N_origin, 1)
dset_origin_sizes[:,] = origin_sizes

# Create a network group
nw_group = h5file.create_group("true_network")
nw_group.attrs["content"] = "graph"
nw_group.attrs["is_directed"] = True
nw_group.attrs["allows_parallel"] = False

# Add vertices
vertices = nw_group.create_dataset(
    "_vertices",
    (1, N_origin + N_destination),
    maxshape=(1, N_origin + N_destination),
    chunks=True,
    compression=3,
    dtype=int,
)
vertices.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
vertices.attrs["coords_mode__vertex_idx"] = "trivial"
vertices[0, :] = np.arange(0, N_origin + N_destination, 1)
vertices.attrs["node_type"] = [0] * N_origin + [1] * N_destination

# Add edges. The network is a complete bipartite graph
edges = nw_group.create_dataset(
    "_edges",
    (1, N_origin * N_destination, 2),
    maxshape=(1, N_origin * N_destination, 2),
    chunks=True,
    compression=3,
)
edges.attrs["dim_names"] = ["dim_name__1", "edge_idx", "vertex_idx"]
edges.attrs["coords_mode__edge_idx"] = "trivial"
edges.attrs["coords_mode__vertex_idx"] = "trivial"
edges[0, :] = np.reshape(
    [
        [[i, j] for i in range(N_origin)]
        for j in range(N_origin, N_origin + N_destination)
    ],
    (N_origin * N_destination, 2),
)

# Edge weights
edge_weights = nw_group.create_dataset(
    "_edge_weights",
    (1, N_origin * N_destination),
    maxshape=(1, N_origin * N_destination),
    chunks=True,
    compression=3,
)
edge_weights.attrs["dim_names"] = ["dim_name__1", "edge_idx"]
edge_weights.attrs["coords_mode__edge_idx"] = "trivial"
edge_weights[0, :] = torch.reshape(network, (N_origin * N_destination,))

# Adjacency matrix: only written out if explicity specified
adjacency_matrix = nw_group.create_dataset(
    "_adjacency_matrix",
    network.shape,
    chunks=True,
    compression=3,
)
adjacency_matrix.attrs["dim_names"] = ["i", "j"]
adjacency_matrix.attrs["coords_mode__i"] = "trivial"
adjacency_matrix.attrs["coords_mode__j"] = "trivial"
adjacency_matrix[:, :] = network

h5file.close()
