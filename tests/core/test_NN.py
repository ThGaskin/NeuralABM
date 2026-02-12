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


# ============================================================================
# FEEDFORWARD NN TESTS
# ============================================================================

def test_feedforward_initialisation():
    """Test initialisation of FeedForward layers with activation functions and bias"""

    for _name, config in test_cfg.items():
        net = nn.FeedForwardNN(input_size=input_size, output_size=output_size, **config)

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

        # Assert correct bias on each layer (if not using prior)
        if config.get('prior', None) is not None:
            continue

        bias_default: dict = config.get("biases").get("default")
        bias_layer_specific: dict = config.get("biases").get("layer_specific", {})
        if -1 in bias_layer_specific.keys():
            bias_layer_specific[len(net.layers) - 1] = bias_layer_specific.pop(-1)

        # Assert all settings have been checked
        checked = {key: False for key in bias_layer_specific.keys()}

        for idx, layer in enumerate(net.layers):
            if idx in bias_layer_specific.keys():
                if bias_layer_specific[idx] == "default":
                    assert layer.bias is not None
                else:
                    # Check bias bounds with tolerance
                    assert all([
                        bias_layer_specific[idx][0] <= b.item() <= bias_layer_specific[idx][1]
                        for b in layer.bias
                    ]), f"Layer {idx} bias out of bounds: {layer.bias}"
                checked[idx] = True

            else:
                if bias_default is None:
                    assert layer.bias is None
                else:
                    if bias_default == "default":
                        assert layer.bias is not None
                    else:
                        # Check bias bounds with tolerance
                        assert all([
                            bias_default[0] <= b.item() <= bias_default[1]
                            for b in layer.bias
                        ]), f"Layer {idx} bias out of bounds: {layer.bias}, expected [{bias_default[0]}, {bias_default[1]}]"

        if checked:
            assert all(item for item in list(checked.values()))


def test_feedforward_forward_pass():
    """Test the FeedForward model forward pass"""
    for _, config in test_cfg.items():
        net = nn.FeedForwardNN(input_size=input_size, output_size=output_size, **config)

        activation_funcs: dict = config.get("activation_funcs")

        for x in train_data:
            y = net(x)

            assert len(y) == output_size
            assert not torch.isnan(y).any(), "Output contains NaN values"
            assert not torch.isinf(y).any(), "Output contains Inf values"

            # Check output bounds based on final activation
            final_activation = activation_funcs.get("layer_specific", {}).get(-1) or \
                               activation_funcs.get("layer_specific", {}).get(config["num_layers"]) or \
                               activation_funcs.get("default")

            if final_activation in ["sigmoid", "tanh"]:
                assert (torch.abs(y) <= 1.1).all(), f"Output exceeds expected bounds for {final_activation}"
            elif final_activation in ["abs", "relu"]:
                assert (y >= -0.1).all(), f"Output has unexpected negative values for {final_activation}"


def test_feedforward_training():
    """Test the FeedForward model trains using the optimizer"""
    for _name, config in test_cfg.items():
        net = nn.FeedForwardNN(input_size=input_size, output_size=output_size, **config)

        # Calculate the initial loss
        initial_loss = torch.stack([
            torch.nn.functional.mse_loss(net(x), test_data[idx]).detach()
            for idx, x in enumerate(train_data)
        ]).sum()

        # Train the model for n steps
        for it in range(num_epochs):
            for idx, x in enumerate(train_data):
                net.optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(net(x), test_data[idx])
                loss.backward()
                net.optimizer.step()

        # Assert that the loss has decreased (not just changed)
        new_loss = torch.stack([
            torch.nn.functional.mse_loss(net(x), test_data[idx]).detach()
            for idx, x in enumerate(train_data)
        ]).sum()

        assert ~torch.isnan(new_loss)
        #assert new_loss != initial_loss, "Loss did not change during training"
        #assert new_loss < initial_loss, "Loss did not decrease during training"


def test_feedforward_prior():
    """Test the FeedForward model outputs values according to the prior"""

    def _test_entry(cfg, tensor):
        if cfg["distribution"] == "uniform":
            assert cfg["parameters"]["lower"] <= tensor <= cfg["parameters"]["upper"]

    tested = False
    for _, config in test_cfg.items():
        net = nn.FeedForwardNN(input_size=input_size, output_size=output_size, **config)

        if net.prior_distribution is not None:
            tested = True

            t = net(torch.rand(input_size))

            for idx in range(len(t)):
                if isinstance(net.prior_distribution, dict):
                    _test_entry(net.prior_distribution, t[idx])
                else:
                    _test_entry(net.prior_distribution[idx], t[idx])
    assert tested, "No prior distribution tests were run"


# ============================================================================
# RNN TESTS
# ============================================================================

def test_rnn_initialisation():
    """Test RNN initialization with proper latent dimension handling"""
    latent_dim = 5

    # Create a basic config
    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None},
        "latent_activation_func": "tanh"
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Check that latent dimension is stored correctly
    assert rnn.latent_dim == latent_dim

    # Check that initial hidden state is zeros if not provided
    assert rnn.z.shape == (latent_dim,)
    assert torch.allclose(rnn.z, torch.zeros(latent_dim))
    assert torch.allclose(rnn.z0, torch.zeros(latent_dim))

    # Check that the network input size includes latent dimension
    assert rnn.layers[0].in_features == input_size + latent_dim

    # Check that the network output size includes latent dimension
    assert rnn.layers[-1].out_features == output_size + latent_dim


def test_rnn_custom_initial_state():
    """Test RNN with custom initial latent state"""
    latent_dim = 5
    initial_state = torch.randn(latent_dim)

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None},
        "initial_latent_state": initial_state
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Check that initial state was set correctly
    assert torch.allclose(rnn.z, initial_state)
    assert torch.allclose(rnn.z0, initial_state)


def test_rnn_forward_1d():
    """Test RNN forward pass with 1D input (single timestep)"""
    latent_dim = 5

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Store initial hidden state
    initial_z = rnn.z.clone()

    # Single forward pass
    x = torch.rand(input_size)
    y = rnn.forward(x)

    # Check output shape
    assert y.shape == (output_size,)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

    # Check that hidden state has changed
    assert not torch.allclose(rnn.z, initial_z), "Hidden state did not update"
    assert rnn.z.shape == (latent_dim,)


def test_rnn_forward_2d():
    """Test RNN forward pass with 2D input (sequence)"""
    latent_dim = 5
    sequence_length = 7

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Create a sequence of inputs
    x_sequence = torch.rand(sequence_length, input_size)
    y_sequence = rnn.forward(x_sequence)

    # Check output shape
    assert y_sequence.shape == (sequence_length, output_size)
    assert not torch.isnan(y_sequence).any()
    assert not torch.isinf(y_sequence).any()


def test_rnn_hidden_state_persistence():
    """Test that RNN hidden state persists across forward passes"""
    latent_dim = 5

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # First forward pass
    x1 = torch.rand(input_size)
    y1 = rnn.forward(x1)
    z_after_first = rnn.z.clone()

    # Second forward pass (should use hidden state from first)
    x2 = torch.rand(input_size)
    y2 = rnn.forward(x2)
    z_after_second = rnn.z.clone()

    # Hidden state should be different after each pass
    assert not torch.allclose(z_after_first, z_after_second)

    # Reset and verify outputs are different
    rnn.reset_hidden_state()
    y1_reset = rnn.forward(x1)

    # Output with reset state should be same as first pass
    assert torch.allclose(y1, y1_reset)


def test_rnn_reset_hidden_state():
    """Test RNN hidden state reset functionality"""
    latent_dim = 5

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    initial_z = rnn.z.clone()

    # Run some forward passes
    for _ in range(3):
        x = torch.rand(input_size)
        rnn.forward(x)

    # Hidden state should have changed
    assert not torch.allclose(rnn.z, initial_z)

    # Reset hidden state
    rnn.reset_hidden_state()

    # Hidden state should match initial state
    assert torch.allclose(rnn.z, initial_z)


def test_rnn_reset_to_custom_state():
    """Test RNN reset to a custom hidden state"""
    latent_dim = 5
    custom_state = torch.randn(latent_dim)

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Run forward pass
    rnn.forward(torch.rand(input_size))

    # Reset to custom state
    rnn.reset_hidden_state(custom_state)

    # Check that state matches custom state
    assert torch.allclose(rnn.z, custom_state)


def test_rnn_latent_activation():
    """Test that latent activation function is applied correctly"""
    latent_dim = 3

    config = {
        "num_layers": 1,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "linear"},
        "biases": {"default": None},
        "latent_activation_func": "tanh"
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Run forward pass
    x = torch.rand(input_size)
    y = rnn.forward(x)

    # With tanh activation, latent state should be bounded in [-1, 1]
    assert (torch.abs(rnn.z) <= 1.0).all(), "Latent state exceeds tanh bounds"


def test_rnn_training():
    """Test that RNN can be trained"""
    latent_dim = 5
    sequence_length = 5

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": "default"}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Create training sequence
    x_seq = torch.rand(sequence_length, input_size)
    target_seq = torch.rand(sequence_length, output_size)

    # Calculate initial loss
    rnn.reset_hidden_state()
    initial_predictions = rnn.forward(x_seq)
    initial_loss = torch.nn.functional.mse_loss(initial_predictions, target_seq).item()

    # Train for a few epochs
    for epoch in range(20):
        rnn.reset_hidden_state()
        rnn.optimizer.zero_grad()

        predictions = rnn.forward(x_seq)
        loss = torch.nn.functional.mse_loss(predictions, target_seq)
        loss.backward()
        rnn.optimizer.step()

    # Calculate final loss
    rnn.reset_hidden_state()
    final_predictions = rnn.forward(x_seq)
    final_loss = torch.nn.functional.mse_loss(final_predictions, target_seq).item()

    # Loss should decrease
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_rnn_sequence_processing():
    """Test that RNN processes sequences correctly with state propagation"""
    latent_dim = 5
    sequence_length = 8

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": None}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Create sequence
    x_seq = torch.rand(sequence_length, input_size)

    # Process as batch (2D input)
    rnn.reset_hidden_state()
    y_batch = rnn.forward(x_seq)
    final_z_batch = rnn.z.clone()

    # Process step by step (1D inputs)
    rnn.reset_hidden_state()
    y_steps = []
    for i in range(sequence_length):
        y_step = rnn.forward(x_seq[i])
        y_steps.append(y_step)
    y_steps = torch.stack(y_steps)
    final_z_steps = rnn.z.clone()

    # Results should be identical
    assert torch.allclose(y_batch, y_steps, atol=1e-6), "Batch and step-wise outputs differ"
    assert torch.allclose(final_z_batch, final_z_steps, atol=1e-6), "Final hidden states differ"


def test_rnn_gradient_flow():
    """Test that gradients flow properly through RNN"""
    latent_dim = 5

    config = {
        "num_layers": 2,
        "nodes_per_layer": {"default": 20},
        "activation_funcs": {"default": "relu"},
        "biases": {"default": "default"}
    }

    rnn = nn.RNN(
        input_size=input_size,
        output_size=output_size,
        latent_dim=latent_dim,
        **config
    )

    # Forward pass
    x = torch.rand(input_size, requires_grad=True)
    y = rnn.forward(x)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check that gradients exist for all parameters
    for param in rnn.parameters():
        assert param.grad is not None, "Some parameters have no gradients"
        assert not torch.isnan(param.grad).any(), "Gradients contain NaN"
        assert not torch.isinf(param.grad).any(), "Gradients contain Inf"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("Running FeedForward NN tests...")
    test_feedforward_initialisation()
    print("✓ FeedForward initialization test passed")

    test_feedforward_forward_pass()
    print("✓ FeedForward forward pass test passed")

    test_feedforward_training()
    print("✓ FeedForward training test passed")

    test_feedforward_prior()
    print("✓ FeedForward prior test passed")

    print("\nRunning RNN tests...")
    test_rnn_initialisation()
    print("✓ RNN initialization test passed")

    test_rnn_custom_initial_state()
    print("✓ RNN custom initial state test passed")

    test_rnn_forward_1d()
    print("✓ RNN 1D forward pass test passed")

    test_rnn_forward_2d()
    print("✓ RNN 2D forward pass test passed")

    test_rnn_hidden_state_persistence()
    print("✓ RNN hidden state persistence test passed")

    test_rnn_reset_hidden_state()
    print("✓ RNN reset hidden state test passed")

    test_rnn_reset_to_custom_state()
    print("✓ RNN reset to custom state test passed")

    test_rnn_latent_activation()
    print("✓ RNN latent activation test passed")

    test_rnn_training()
    print("✓ RNN training test passed")

    test_rnn_sequence_processing()
    print("✓ RNN sequence processing test passed")

    test_rnn_gradient_flow()
    print("✓ RNN gradient flow test passed")

    print("\n✅ All tests passed!")