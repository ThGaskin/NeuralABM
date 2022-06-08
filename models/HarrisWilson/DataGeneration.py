import os
import pandas as pd
import torch

from .ABM import HarrisWilsonABM

""" Load a dataset or generate synthetic data on which to train the neural net """

def get_HW_data(cfg, *, out_dir: str):

    """ Gets the data for the Harris-Wilson model. If no path to a dataset is passed, synthetic data is generated using
    the config settings

    :param cfg: the data configuration
    :param out_dir: the output path to which to write the synthetic data, if generated
    :return: the origin zone sizes, the training data, and the network
    """

    if 'load_from_dir' in cfg.keys():
        print("\nLoading data from directory ...")

        # Load the origin zone sizes and reshape
        or_sizes = torch.tensor(pd.read_csv(cfg['load_from_dir'] + '/origin_sizes.csv', header=0, index_col=0).values,
                                dtype=torch.float)
        N = len(or_sizes)
        or_sizes = torch.reshape(or_sizes, (N, 1))

        # Load the destination zone sizes and reshape
        dest_sizes = torch.tensor(pd.read_csv(cfg['load_from_dir'] + '/training_data.csv', header=0, index_col=0).values,
                                  dtype=torch.float)
        M = dest_sizes.shape[1]
        dest_sizes = torch.reshape(dest_sizes, (len(dest_sizes), M, 1))

        # Load the network and reshape
        network = torch.tensor(pd.read_csv(cfg['load_from_dir'] + '/network.csv', header=0, index_col=0).values, dtype=torch.float)
        network = torch.reshape(network, (N, M))

    else:
        print("\nGenerating synthetic data ...")

        # Create directory for synthetic data
        os.makedirs(os.path.join(out_dir, 'synthetic_data'))
        out_dir = out_dir + '/synthetic_data/'
        data_cfg = cfg['synthetic_data']

        # Generate origin zone sizes, initial destination zone sizes, and the network.
        or_sizes = torch.abs(torch.normal(0.1, 0.01, size=(data_cfg['N_origin'], 1)))

        init_dest_sizes = torch.abs(torch.normal(0.1, 0.01, size=(data_cfg['N_destination'], 1)))
        network = torch.exp(-1 * torch.rand(size = (data_cfg['N_origin'], data_cfg['N_destination'])))

        # Save the origin zones and network
        pd.DataFrame(or_sizes, columns=['size']).to_csv(out_dir + '/origin_sizes.csv', index_label='origin_zone')
        pd.DataFrame(network).to_csv(out_dir + '/network.csv', index_label='row: origin_zone/col: destination_zone')

        # Extract the underlying parameters from the config
        true_parameters = {'alpha': data_cfg['alpha'], 'beta': data_cfg['beta'], 'kappa': data_cfg['kappa']}

        # Initialise the ABM
        ABM = HarrisWilsonABM(origin_sizes=or_sizes, network=network, true_parameters=true_parameters,
                              M=data_cfg['N_destination'], epsilon=data_cfg['epsilon'], dt=data_cfg['dt'],
                              sigma=data_cfg['sigma'])

        # Run the ABM for n iterations and save the entire time series
        dest_sizes_ts = ABM.run(init_data=init_dest_sizes, input_data=None, n_iterations=data_cfg['num_steps'],
                                generate_time_series=True, requires_grad=False)
        pd.DataFrame(torch.flatten(dest_sizes_ts, start_dim=1)).to_csv(out_dir + '/time_series.csv',
                                                                       index_label='row: time/col: destination_zone')

        # Extract the training data from the time series data and save
        dest_sizes = dest_sizes_ts[-data_cfg['training_data_size']:]
        pd.DataFrame(torch.flatten(dest_sizes, start_dim=1)).to_csv(out_dir + '/training_data.csv',
                                                                    index_label='row: time/col: destination_zone')

    return or_sizes, dest_sizes, network
