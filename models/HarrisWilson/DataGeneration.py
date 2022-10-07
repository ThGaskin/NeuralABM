import h5py as h5
import numpy as np
import pandas as pd
import torch
import logging
from typing import Tuple

from .ABM import HarrisWilsonABM

""" Load a dataset or generate synthetic data on which to train the neural net """

log = logging.getLogger(__name__)

def load_from_dir(dir) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    """Loads Harris-Wilson data from a directory.

    :returns the origin sizes, network, and the time series
    """

    log.note('   Loading data ...')

    # If data is to be loaded, check whether a single h5 file, a folder containing csv files, or
    # a dictionary pointing to specific csv files has been passed.
    if isinstance(dir, str):

        # If data is in h5 format
        if dir.lower().endswith('.h5'):
            with h5.File(dir, "r") as f:
                origins = np.array(f['HarrisWilson']['origin_sizes'])[0]
                training_data = np.array(f['HarrisWilson']['training_data'])
                nw = np.array(f['network']['_edge_weights'])

        # If data is a folder, load csv files
        else:
            origins = pd.read_csv(dir + '/origin_sizes.csv', header=0, index_col=0).to_numpy()
            training_data = pd.read_csv(dir + '/training_data.csv', header=0, index_col=0).to_numpy()
            nw = pd.read_csv(dir + '/network.csv', header=0, index_col=0).to_numpy()

    # If a dictionary is passed, load data from individual locations
    elif isinstance(dir, dict):

        origins = pd.read_csv(dir['origin_zones'], header=0, index_col=0).to_numpy()
        training_data = pd.read_csv(dir['destination_zones'], header=0, index_col=0).to_numpy()
        nw = pd.read_csv(dir['network'], header=0, index_col=0).to_numpy()

    # Reshape the origin zone sizes
    or_sizes = torch.tensor(origins, dtype=torch.float)
    N_origin = len(or_sizes)
    or_sizes = torch.reshape(or_sizes, (N_origin, 1))

    # Reshape the time series of the destination zone sizes
    time_series = torch.tensor(training_data, dtype=torch.float)
    N_destination = training_data.shape[1]
    time_series = torch.reshape(time_series, (len(time_series), N_destination, 1))

    # Reshape the network
    network = torch.reshape(torch.tensor(nw, dtype=torch.float), (N_origin, N_destination))

    # Return all three datasets
    return or_sizes, time_series, network

def generate_synthetic_data(*, cfg) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    """Generates synthetic Harris-Wilson using a numerical solver.

    :param cfg: the configuration file
    :returns the origin sizes, network, and the time series
    """

    log.note("   Generating synthetic data ...")

    # Get run configuration properties
    data_cfg = cfg['synthetic_data']
    N_origin, N_destination = data_cfg['N_origin'], data_cfg['N_destination']
    num_steps = data_cfg['num_steps']

    # Generate the initial origin sizes
    or_sizes = torch.abs(torch.normal(0.1, 0.01, size=(N_origin, 1)))

    # Generate the edge weights
    network = torch.exp(-1 * torch.abs(torch.normal(data_cfg['init_weights']['mean'],
                                                    data_cfg['init_weights']['std'],
                                                    size = (N_origin, N_destination))))

    # Generate the initial destination zone sizes
    init_dest_sizes = torch.abs(torch.normal(0.1, 0.01, size=(data_cfg['N_destination'], 1)))

    # Extract the underlying parameters from the config
    true_parameters = {'alpha': data_cfg['alpha'], 'beta': data_cfg['beta'], 'kappa': data_cfg['kappa'],
                       'sigma': data_cfg['sigma']}

    # Initialise the ABM
    ABM = HarrisWilsonABM(origin_sizes=or_sizes, network=network, true_parameters=true_parameters,
                          M=data_cfg['N_destination'], epsilon=data_cfg['epsilon'], dt=data_cfg['dt'],
                          device='cpu')

    # Run the ABM for n iterations, generating the entire time series
    dset_sizes_ts = ABM.run(init_data=init_dest_sizes, input_data=None, n_iterations=num_steps,
                            generate_time_series=True, requires_grad=False)

    # Return all three
    return or_sizes, dset_sizes_ts, network

def get_HW_data(cfg, h5file: h5.File, h5group: h5.Group, *, device: str):

    """ Gets the data for the Harris-Wilson model. If no path to a dataset is passed, synthetic data is generated using
    the config settings

    :param cfg: the data configuration
    :param h5file: the h5 File to use. Needed to add a network group.
    :param h5group: the h5 Group to write data to
    :return: the origin zone sizes, the training data, and the network
    """

    data_dir = cfg.pop('load_from_dir', None)

    # Get the origin sizes, time series, and network data
    or_sizes, dest_sizes, network = load_from_dir(data_dir) if data_dir is not None else generate_synthetic_data(cfg=cfg)

    N_origin = or_sizes.shape[0]
    N_destination = dest_sizes.shape[1]

    # Only save individual time frames
    synthetic_data = cfg.pop('synthetic_data', None)
    if synthetic_data is not None:
      write_start = synthetic_data.pop('write_start', 0)
      write_every = synthetic_data.pop('write_every', 1)
      time_series = dest_sizes[write_start::write_every]
    else:
      time_series = dest_sizes

    # If time series has a single frame, double it to enable visualisation.
    # This does not affect the training data
    training_data_size = cfg.pop('training_data_size', len(time_series))
    if len(time_series) == 1:
      time_series = torch.stack([time_series, time_series])

    # Set up dataset for complete synthetic time series
    dset_time_series = h5group.create_dataset(
        "time_series",
        (len(time_series), N_destination),
        maxshape=(len(time_series), N_destination),
        chunks=True,
        compression=3,
    )
    dset_time_series.attrs['dim_names'] = ['time', 'zone_id']
    dset_time_series.attrs['coords_mode__time'] = 'start_and_step'
    dset_time_series.attrs['coords__time'] = [write_start, write_every]
    dset_time_series.attrs['coords_mode__zone_id'] = 'values'
    dset_time_series.attrs['coords__zone_id'] = np.arange(N_origin, N_origin + N_destination, 1)

    # Write the time series data
    dset_time_series[:, :] = torch.flatten(time_series, start_dim=1)

    # Training time series
    dset_training_data = h5group.create_dataset(
        "training_data",
        (training_data_size, N_destination),
        maxshape=(training_data_size, N_destination),
        chunks=True,
        compression=3
    )
    dset_training_data.attrs['dim_names'] = ['time', 'zone_id']
    dset_training_data.attrs['coords_mode__time'] = 'trivial'
    dset_training_data.attrs['coords_mode__zone_id'] = 'values'
    dset_training_data.attrs['coords__zone_id'] = np.arange(N_origin, N_origin + N_destination, 1)

    # Extract the training data from the time series data and save
    training_data = dest_sizes[-training_data_size:]
    dset_training_data[:, :] = torch.flatten(training_data, start_dim=1)

    # Set up chunked dataset to store the state data in
    # Origin zone sizes
    dset_origin_sizes = h5group.create_dataset(
        "origin_sizes",
        (1, N_origin),
        maxshape=(1, N_origin),
        chunks=True,
        compression=3,
    )
    dset_origin_sizes.attrs['dim_names'] = ['dim_name__0', 'zone_id']
    dset_origin_sizes.attrs["coords_mode__zone_id"] = "values"
    dset_origin_sizes.attrs["coords__zone_id"] = np.arange(0, N_origin, 1)
    dset_origin_sizes[0, :] = torch.flatten(or_sizes)

    # Create a network group
    nw_group = h5file.create_group('network')
    nw_group.attrs['content'] = 'graph'
    nw_group.attrs['is_directed'] = True
    nw_group.attrs['allows_parallel'] = False

    # Add vertices
    vertices = nw_group.create_dataset(
        "_vertices",
        (1, N_origin + N_destination),
        maxshape=(1, N_origin + N_destination),
        chunks=True,
        compression=3,
        dtype=int
    )
    vertices.attrs['dim_names'] = ['dim_name__0', 'vertex_idx']
    vertices.attrs["coords_mode__vertex_idx"] = "trivial"
    vertices[0, :] = np.arange(0, N_origin + N_destination, 1)
    vertices.attrs['node_type'] = [0] * N_origin + [1] * N_destination

    # Add edges. The network is a complete bipartite graph
    # TODO: allow more general network topologies?
    edges = nw_group.create_dataset(
        "_edges",
        (1, N_origin * N_destination, 2),
        maxshape=(1, N_origin * N_destination, 2),
        chunks=True,
        compression=3,
    )
    edges.attrs['dim_names'] = ['dim_name__1', 'edge_idx', 'vertex_idx']
    edges.attrs["coords_mode__edge_idx"] = "trivial"
    edges.attrs["coords_mode__vertex_idx"] = "trivial"
    edges[0, :] = np.reshape(
        [[[i, j] for i in range(N_origin)] for j in range(N_origin, N_origin+N_destination)],
        (N_origin*N_destination, 2))

    # Edge weights
    edge_weights = nw_group.create_dataset(
        "_edge_weights",
        (1, N_origin * N_destination),
        maxshape=(1, N_origin * N_destination),
        chunks=True,
        compression=3,
    )
    edge_weights.attrs['dim_names'] = ['dim_name__1', 'edge_idx']
    edge_weights.attrs["coords_mode__edge_idx"] = "trivial"
    edge_weights[0, :] = torch.reshape(network, (N_origin * N_destination, ))

    return or_sizes.to(device), training_data.to(device), network.to(device)
