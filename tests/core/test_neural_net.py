import sys
from os.path import dirname as up

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

        # Assert correct number of layers
        assert (
            len(net.layers)
            == config["num_layers"] + 1  # input layer + number of hidden layers
        )

        # Assert correct input size
        assert net.layers[0].in_features == input_size

        # Assert correct output size
        assert net.layers[-1].out_features == output_size

        # Assert correct dimensions of hidden layers
        layer_cfg: dict = config["nodes_per_layer"]
        layer_specific_cfg: dict = layer_cfg.get("layer_specific", {})
        if -1 in layer_specific_cfg.keys():
            layer_specific_cfg[len(net.layers) - 2] = layer_specific_cfg.pop(-1)
        hidden_layers = net.layers[1:]

        # Assert all settings have been checked
        checked = {key: False for key in layer_specific_cfg.keys()}

        # Check layers have correct number of nodes
        for idx, layer in enumerate(hidden_layers):

            if idx in layer_specific_cfg.keys():
                assert layer.in_features == layer_specific_cfg[idx]
                checked[idx] = True
            else:
                assert layer.in_features == layer_cfg["default"]

            if idx != len(net.layers) - 2:
                assert layer.out_features == net.layers[idx + 2].in_features
            elif idx == len(net.layers) - 2:
                assert layer.out_features == output_size

        if checked:
            assert all(item for item in list(checked.values()))
        del checked

        # Assert correct bias on each layer
        bias_default: dict = config.get("biases").get("default")
        bias_layer_specific: dict = config.get("biases").get("layer_specific", {})
        if -1 in bias_layer_specific.keys():
            bias_layer_specific[len(net.layers) - 1] = bias_layer_specific.pop(-1)

        # Assert all settings have been checked
        checked = {key: False for key in bias_layer_specific.keys()}

        for idx, layer in enumerate(net.layers):

            if idx in bias_layer_specific.keys():
                assert [
                    bias_layer_specific[idx][0] <= b <= bias_layer_specific[idx][1]
                    for b in layer.bias
                ]
                checked[idx] = True

            else:
                if bias_default is None:
                    assert layer.bias is None
                else:
                    assert [bias_default[0] <= b <= bias_default[1] for b in layer.bias]

        if checked:
            assert all(item for item in list(checked.values()))


# Test the model forward pass
def test_forward_pass():
    for _, config in test_cfg.items():

        net = nn.NeuralNet(input_size=input_size, output_size=output_size, **config)

        activation_funcs: dict = config.get("activation_funcs")

        for x in train_data:
            y = net(x)

            assert len(y) == output_size

            if list(activation_funcs.values())[-1] in ["sigmoid", "tanh"]:
                assert (torch.abs(y) <= 1).all()
            elif activation_funcs in ["abs", "sigmoid"]:
                assert (y >= 0).all()


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
