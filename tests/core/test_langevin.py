import tempfile
import torch
import h5py as h5
import pytest
from pathlib import Path

import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path

# Import the module to test
sys.path.insert(0, up(up(up(__file__))))

include = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="include"
)
from include.langevin import MetropolisAdjustedLangevin, pSGLD

# ============================================================================
# pSGLD OPTIMIZER TESTS
# ============================================================================

def test_psgld_initialization():
    """Test pSGLD optimizer initializes correctly"""
    params = [torch.randn(10, 5, requires_grad=True)]

    optimizer = pSGLD(params, lr=0.01, beta=0.99, Lambda=1e-15)

    assert optimizer is not None
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.01
    assert optimizer.param_groups[0]['beta'] == 0.99
    assert optimizer.param_groups[0]['Lambda'] == 1e-15


def test_psgld_invalid_params():
    """Test pSGLD raises errors for invalid parameters"""
    params = [torch.randn(10, 5, requires_grad=True)]

    # Invalid learning rate
    with pytest.raises(ValueError, match="Invalid learning rate"):
        pSGLD(params, lr=-0.01)

    # Invalid epsilon
    with pytest.raises(ValueError, match="Invalid epsilon value"):
        pSGLD(params, Lambda=-1e-5)

    # Invalid weight decay
    with pytest.raises(ValueError, match="Invalid weight_decay value"):
        pSGLD(params, weight_decay=-0.1)

    # Invalid beta
    with pytest.raises(ValueError, match="Invalid beta value"):
        pSGLD(params, beta=-0.5)


def test_psgld_step():
    """Test pSGLD optimizer can perform a step"""
    # Simple quadratic loss: f(x) = x^2
    x = torch.tensor([2.0], requires_grad=True)
    optimizer = pSGLD([x], lr=0.01)

    # Compute gradient
    loss = x ** 2
    loss.backward()

    # Store initial value
    x_initial = x.data.clone()

    # Perform optimization step
    G = optimizer.step()

    # Check that parameter changed
    assert not torch.equal(x.data, x_initial)

    # Check that G was returned
    assert G is not None

    # Check that value is positive (due to abs_() at end)
    assert (x.data >= 0).all()


def test_psgld_state_initialization():
    """Test pSGLD correctly initializes state"""
    x = torch.randn(5, 3, requires_grad=True)
    optimizer = pSGLD([x], lr=0.01, centered=True)

    # Before step, state should be empty
    assert len(optimizer.state) == 0

    # Compute gradient and step
    loss = (x ** 2).sum()
    loss.backward()
    optimizer.step()

    # After step, state should be initialized
    assert len(optimizer.state) == 1
    state = optimizer.state[x]

    assert 'step' in state
    assert state['step'] == 1
    assert 'V' in state
    assert state['V'].shape == x.shape
    assert 'grad_avg' in state  # Because centered=True
    assert state['grad_avg'].shape == x.shape


def test_psgld_sparse_gradients():
    """Test pSGLD raises error for sparse gradients"""
    x = torch.randn(10, requires_grad=True)
    optimizer = pSGLD([x], lr=0.01)

    # Create sparse gradient
    indices = torch.tensor([0, 2, 4])
    values = torch.tensor([1.0, 2.0, 3.0])
    x.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), values, (10,))

    # Should raise error
    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        optimizer.step()


# ============================================================================
# METROPOLIS-ADJUSTED LANGEVIN TESTS
# ============================================================================

def test_mal_initialization():
    """Test MetropolisAdjustedLangevin initializes correctly"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.randn(10, 5)
            init_guess = torch.randn(10, 5)

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                lr_final=0.0001,
                max_itr=100,
                h5File=f
            )

            assert mal is not None
            assert mal.time == 0
            assert mal.lr == 0.01
            assert mal.lr_final == 0.0001
            assert mal.max_itr == 100

            # Check x tensors initialized
            assert mal.x[0].shape == init_guess.shape
            assert mal.x[1].shape == init_guess.shape
            assert torch.equal(mal.x[0].data, init_guess)

            # Check h5 datasets created
            assert 'langevin_data' in f
            assert 'loss' in f['langevin_data']
            assert 'time' in f['langevin_data']
    finally:
        tmp_path.unlink()


def test_mal_decay_function():
    """Test learning rate decay function"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.randn(5, 3)
            init_guess = torch.randn(5, 3)

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                lr_final=0.0001,
                max_itr=100,
                h5File=f
            )

            # Test decay function
            lr_0 = mal.lr_fn(0)
            lr_50 = mal.lr_fn(50)
            lr_100 = mal.lr_fn(100)

            # Learning rate should decrease over time
            assert lr_0 >= lr_50 >= lr_100

            # Initial lr should be close to specified lr
            assert abs(lr_0 - 0.01) < 0.001

            # Final lr should be close to specified lr_final
            assert abs(lr_100 - 0.0001) < 0.01
    finally:
        tmp_path.unlink()


def test_mal_proposal_distribution():
    """Test proposal distribution calculation"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.randn(5, 3)
            init_guess = torch.randn(5, 3)

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                h5File=f
            )

            # Set some gradients
            mal.grad[0] = torch.randn_like(init_guess)
            mal.grad[1] = torch.randn_like(init_guess)

            # Calculate proposal distributions
            prop_0 = mal.proposal_dist(0)
            prop_1 = mal.proposal_dist(1)

            # Should return tensors
            assert isinstance(prop_0, torch.Tensor)
            assert isinstance(prop_1, torch.Tensor)

            # Should be scalar values
            assert prop_0.numel() == 1
            assert prop_1.numel() == 1
    finally:
        tmp_path.unlink()


def test_mal_sample_with_loss_function():
    """Test sampling with a simple loss function"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.tensor([1.0, 2.0, 3.0])
            init_guess = torch.tensor([0.0, 0.0, 0.0])

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                lr_final=0.001,
                max_itr=10,
                h5File=f
            )

            # Define a simple loss function (MSE with true_data)
            def loss_fn(x):
                return ((x - mal.true_data) ** 2).sum()

            mal.loss_function = loss_fn

            # Compute initial loss and gradient
            mal.loss[0] = loss_fn(mal.x[0])
            mal.grad[0] = torch.autograd.grad(
                mal.loss[0], [mal.x[0]], create_graph=False
            )[0]

            # Force accept first sample
            sample, loss_val = mal.sample(force_accept=True)

            # Check return types
            assert isinstance(sample, torch.Tensor)
            assert isinstance(loss_val, float)

            # Check that time incremented
            assert mal.time == 1

            # Check that sample is positive (due to abs_() in pSGLD)
            assert (sample >= 0).all()
    finally:
        tmp_path.unlink()


def test_mal_write_loss():
    """Test writing loss to h5 file"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.randn(5)
            init_guess = torch.randn(5)

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                write_start=0,
                write_every=1,
                h5File=f
            )

            # Set a loss value
            mal.loss[0] = torch.tensor([0.5])
            mal.time = 1

            # Write loss
            initial_size = mal.dset_loss.shape[0]
            mal.write_loss()

            # Check that dataset grew
            assert mal.dset_loss.shape[0] == initial_size + 1
            assert mal.dset_loss[-1] == 0.5
    finally:
        tmp_path.unlink()


def test_mal_write_time():
    """Test writing time to h5 file"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.randn(5)
            init_guess = torch.randn(5)

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                h5File=f
            )

            # Write time
            initial_size = mal.dset_time.shape[0]
            mal.write_time(123.45)

            # Check that dataset grew
            assert mal.dset_time.shape[0] == initial_size + 1
            assert mal.dset_time[-1] == 123.45
    finally:
        tmp_path.unlink()


def test_mal_lr_decay():
    """Test learning rate decay updates optimizer"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with h5.File(tmp_path, 'w') as f:
            true_data = torch.randn(5)
            init_guess = torch.randn(5)

            mal = MetropolisAdjustedLangevin(
                true_data=true_data,
                init_guess=init_guess,
                lr=0.01,
                lr_final=0.0001,
                max_itr=100,
                h5File=f
            )

            # Initial learning rate
            initial_lr = mal.optim.param_groups[0]['lr']

            # Advance time and decay
            mal.time = 50
            mal.lr_decay()

            # Learning rate should have decreased
            new_lr = mal.optim.param_groups[0]['lr']
            assert new_lr < initial_lr
            assert new_lr == mal.lr_fn(50)
    finally:
        tmp_path.unlink()