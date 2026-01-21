import argparse
import numpy
import os
import pickle
import torch
import tqdm

from typing import Literal, Union

# Local imports
from parameters_only import get_params_NN
from hybrid_1 import get_hybrid_1_NN
from hybrid_2 import get_hybrid_2_NN
from black_box import get_bb_NN
from utils import epoch

def train(*,
          L: int = 30,
          n: int = 1,
          key: Literal['params', 'hybrid_1', 'hybrid_2', 'bb'] = 'params',
          seed: int = None,
          dt: Union[float, torch.Tensor] = 0.2,
          training_data: str,
          recursive: bool = False,
          out_dir: str = None,
          N_epochs: int = None
):
    # Set the seed, if passed
    if seed is not None:
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    # Load the training data and set the initial condition
    Y = torch.load(training_data, weights_only=True)
    y0 = Y[:, 0, :]

    # Set up a dictionary with all the required data
    data = {
        'Y_target': Y[:n],
        'X_input': Y[:n, :(L+1)].flatten(start_dim=1),
        'loss': []
    }

    # Add dataset identifier if sweeping over more than one training dataset
    if n > 1:
        if recursive:
            data['Y_input'] = Y[:n, :L]
            data['z'] = Y[:n, :3, :2].flatten(start_dim=1)
        else:
            data['Y_input'] = torch.cat([
                Y[:n, :L, :], Y[:n, :3, :2].flatten(start_dim=1)[:, None, :].repeat(1, L, 1)
            ], dim=2)
    else:
        data['Y_input'] = Y[:n, :(L+1)]

    # Get the neural network
    if key == 'params':
        data['NN'] = get_params_NN(input_size=data['X_input'].shape[1], z=6 if n>1 else 0)
    elif key == 'hybrid_1':
        data['NN'] = get_hybrid_1_NN(input_size=data['X_input'].shape[1], z=6 if n>1 else 0)
    elif key == 'hybrid_2':
        data['NN'] = get_hybrid_2_NN(z=6 if n>1 else 0)
    elif key == 'bb':
        data['NN'] = get_bb_NN(z=6 if n>1 else 0)

    # Save the trained network and loss evolution to a folder
    if out_dir is not None:
        path_name = f"{key}__n_{n}__L_{L}"
        if recursive:
            path_name += "__recursive"
        if seed is not None:
            path_name += f"__seed_{seed}"
        path_name = os.path.expanduser(os.path.join(out_dir, path_name))
        os.makedirs(path_name, exist_ok=True)
        if key !='hybrid_1':
            data['NN'].load_state_dict(torch.load(f"{path_name}/NN.pt", weights_only=True))
            data['NN'].eval()
        else:
            data['NN']['const_params'].load_state_dict(torch.load(f"{path_name}/NN_const_params.pt", weights_only=True))
            data['NN']['time_dep_params'].load_state_dict(torch.load(f"{path_name}/NN_time_dep_params.pt", weights_only=True))
            data['NN']['const_params'].eval()
            data['NN']['time_dep_params'].eval()
        with open(f"{path_name}/loss.pickle", "rb") as f:
            loss = pickle.load(f)

    # Train for N_epochs
    if N_epochs is None:
        N_epochs = 10000 if key == 'params' else 20000

    N_epochs -= len(loss)
    print(f'Remaining: {N_epochs}')
    for i in tqdm.tqdm(range(N_epochs)):
        epoch(key=key, NN=data['NN'], X_input=data['X_input'], Y_target=data['Y_target'][:, :L], Y_input=data['Y_input'],
              dt=dt, y0=y0, t_span=(0, (L-1)*dt), recursive=recursive, z=data.get('z', None), loss_array=data['loss'])

        # Store the results every 100 epochs
        if ((i > 0 and i % 100 == 0) or i == N_epochs-1) and out_dir is not None:
            if key != 'hybrid_1':
                torch.save(data['NN'].state_dict(), f"{path_name}/NN.pt")
            else:
                torch.save(data['NN']['const_params'].state_dict(), f"{path_name}/NN_const_params.pt")
                torch.save(data['NN']['time_dep_params'].state_dict(), f"{path_name}/NN_time_dep_params.pt")
            with open(f"{path_name}/loss.pickle", "wb") as f:
                pickle.dump(data['loss'], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=30, help="Length of training time series")
    parser.add_argument("--n", type=int, default=1, help="Number of training datasets to use")
    parser.add_argument("--key", type=str, default='params', help="Model to use")
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--recursive", action="store_true", help="Whether to generate predictions recursively")
    parser.add_argument("--seed", type=int, default=None, help="Set the seed")
    parser.add_argument("--N_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    args = parser.parse_args()

    train(key=args.key, n=args.n, L=args.L, training_data=args.training_data, recursive=args.recursive, seed=args.seed,
          N_epochs=args.N_epochs, out_dir=args.out_dir)