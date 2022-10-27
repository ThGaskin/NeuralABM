import sys
from os.path import dirname as up
from typing import Sequence

import torch
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

nn = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="include.neural_net"
)

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/neural_net.yml")
test_cfg = load_yml(CFG_FILENAME)

# Generate some training data
test_data, train_data = torch.rand((10, 10), dtype=torch.float), torch.rand(
    (3, 10), dtype=torch.float
)
input_size = output_size = test_data.shape[1]
num_epochs = 10

# Test initialisation of the layers with activation functions and bias
def test_initialisation():
    for _, config in test_cfg.items():

        net = nn.NeuralNet(input_size=input_size, output_size=output_size, **config)

        assert net
        assert (
            len(net.layers) == config["num_layers"] + 1
        )  # input layer + number of hidden layers
        assert net.layers[0].in_features == input_size
        assert net.layers[-1].out_features == output_size

        biases = config.pop("biases", None)

        for idx, layer in enumerate(net.layers):

            # Check whether layer has bias. If bias, check that bias values are within given interval
            if biases is None:
                assert layer.bias is None

            elif isinstance(biases, Sequence):
                assert [
                    biases[0] <= b <= biases[1]
                    for b in layer.bias
                ]
            else:
                if idx in biases.keys():
                    if biases[idx] is None:
                        assert layer.bias is None
                    else:
                        assert [
                            biases[idx][0] <= b <= biases[idx][1]
                            for b in layer.bias
                        ]
                else:
                    assert layer.bias is None

            # Check the layers have correct dimensions
            if idx == 0:
                assert layer.in_features == input_size
            elif idx == len(net.layers) - 1:
                assert layer.out_features == output_size
            else:
                assert (
                    layer.in_features == layer.out_features == config["nodes_per_layer"]
                )


# Test the model forward pass
def test_forward_pass():
    for _, config in test_cfg.items():

        net = nn.NeuralNet(input_size=input_size, output_size=output_size, **config)

        activation_funcs = config.pop("activation_funcs", None)

        for x in train_data:
            y = net(x)
            assert len(y) == output_size

            if activation_funcs and isinstance(activation_funcs, str):
                if activation_funcs in ["abs", "sigmoid"]:
                    assert (y >= 0).all()

            elif activation_funcs and isinstance(activation_funcs, dict):
                if list(activation_funcs.values())[-1] in ["sigmoid", "tanh"]:
                    assert (torch.abs(y) <= 1).all()


# Test the model trains using the optimizer
def test_training():
    for _, config in test_cfg.items():

        net = nn.NeuralNet(input_size=input_size, output_size=output_size, **config)

        for it in range(num_epochs):

            for idx, x in enumerate(train_data):

                # Perform a single training step
                loss = torch.tensor(0.0, requires_grad=True)
                net.optimizer.zero_grad()
                y = net(x)
                loss = torch.nn.functional.mse_loss(y, test_data[idx])
                loss.backward()
                net.optimizer.step()

                # Assert that the model is nonzero
                previous_loss = loss.detach().numpy()
                assert previous_loss > 0
                del loss

                # Repeat the training step on the same batch and assert that the loss has changed, meaning
                # the internal parameters have changed
                loss = torch.tensor(0.0, requires_grad=True)
                net.optimizer.zero_grad()
                y = net(x)
                loss = torch.nn.functional.mse_loss(y, test_data[idx])
                loss.backward()
                net.optimizer.step()
                current_loss = loss.detach().numpy()

                assert current_loss != previous_loss
