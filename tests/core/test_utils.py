import sys
from builtins import *
from os.path import dirname as up
import tempfile
from pathlib import Path

import pytest
import torch
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

utils = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include.utils")

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/test_utils.yml")
test_cfg = load_yml(CFG_FILENAME)


# ============================================================================
# LOAD_CONFIG TESTS
# ============================================================================

def test_load_config_basic():
    """Test loading a basic config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
root_model_name: TestModel
TestModel:
  param1: value1
  param2: 42
other_setting: xyz
""")
        config_path = f.name

    try:
        cfg, model_cfg = utils.load_config(config_path)

        assert cfg is not None
        assert model_cfg is not None
        assert cfg['root_model_name'] == 'TestModel'
        assert model_cfg['param1'] == 'value1'
        assert model_cfg['param2'] == 42
        assert cfg['other_setting'] == 'xyz'
    finally:
        Path(config_path).unlink()


def test_load_config_default_model_name():
    """Test loading config with default model name (SIR)"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
SIR:
  beta: 0.5
  gamma: 0.1
other_param: test
""")
        config_path = f.name

    try:
        cfg, model_cfg = utils.load_config(config_path)

        # Should default to 'SIR' when root_model_name not specified
        assert model_cfg['beta'] == 0.5
        assert model_cfg['gamma'] == 0.1
    finally:
        Path(config_path).unlink()


def test_load_config_nested_structure():
    """Test loading config with nested structure"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
root_model_name: ComplexModel
ComplexModel:
  layer1:
    sublayer1: value1
    sublayer2: value2
  layer2: [1, 2, 3]
""")
        config_path = f.name

    try:
        cfg, model_cfg = utils.load_config(config_path)

        assert 'layer1' in model_cfg
        assert model_cfg['layer1']['sublayer1'] == 'value1'
        assert model_cfg['layer2'] == [1, 2, 3]
    finally:
        Path(config_path).unlink()


def test_load_config_file_not_found():
    """Test that load_config raises error for non-existent file"""
    with pytest.raises(FileNotFoundError):
        utils.load_config('/nonexistent/path/to/config.yml')


# ============================================================================
# SET_DEFAULT_DEVICE TESTS
# ============================================================================

def test_set_default_device_cpu():
    """Test setting device to CPU"""
    cfg = {'device': 'cpu'}
    device = utils.set_default_device(cfg)

    assert device == torch.device('cpu')
    assert torch.get_default_device().type == 'cpu'


def test_set_default_device_default():
    """Test default device when not specified in config"""
    cfg = {}
    device = utils.set_default_device(cfg)

    assert device == torch.device('cpu')


def test_set_default_device_with_num_threads():
    """Test setting number of threads"""
    original_threads = torch.get_num_threads()

    cfg = {'device': 'cpu', 'num_threads': 2}
    device = utils.set_default_device(cfg)

    assert device == torch.device('cpu')
    assert torch.get_num_threads() == 2

    # Restore original
    torch.set_num_threads(original_threads)


def test_set_default_device_without_num_threads():
    """Test that num_threads is optional"""
    cfg = {'device': 'cpu'}
    device = utils.set_default_device(cfg)

    assert device == torch.device('cpu')
    # Should not raise error for missing num_threads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_default_device_cuda():
    """Test setting device to CUDA (if available)"""
    cfg = {'device': 'cuda'}
    device = utils.set_default_device(cfg)

    assert device.type == 'cuda'

    torch.set_default_device('cpu')


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_set_default_device_mps():
    """Test setting device to MPS/Metal (if available)"""
    cfg = {'device': 'mps'}
    device = utils.set_default_device(cfg)

    assert device.type == 'mps'

    torch.set_default_device('cpu')


# ============================================================================
# RANDOM_TENSOR TESTS (Enhanced)
# ============================================================================

def test_random_tensor_uniform_dict():
    """Test uniform distribution with dict config"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 0.0, 'upper': 1.0}
    }

    tensor = utils.random_tensor(cfg, size=(100,))

    assert tensor.shape == (100,)
    assert (tensor >= 0.0).all()
    assert (tensor <= 1.0).all()


def test_random_tensor_uniform_bounds():
    """Test uniform distribution with various bounds"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': -5.0, 'upper': 5.0}
    }

    tensor = utils.random_tensor(cfg, size=(100,))

    assert (tensor >= -5.0).all()
    assert (tensor <= 5.0).all()


def test_random_tensor_uniform_equal_bounds():
    """Test uniform distribution with equal lower and upper bounds"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 3.0, 'upper': 3.0}
    }

    tensor = utils.random_tensor(cfg, size=(10,))

    # Should all be approximately 3.0
    assert torch.allclose(tensor, torch.tensor(3.0), atol=1e-6)


def test_random_tensor_uniform_invalid_bounds():
    """Test that invalid bounds raise ValueError"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 5.0, 'upper': 1.0}  # lower > upper
    }

    with pytest.raises(ValueError, match="Upper bound must be greater or equal to lower bound"):
        utils.random_tensor(cfg, size=(10,))


def test_random_tensor_normal_dict():
    """Test normal distribution with dict config"""
    cfg = {
        'distribution': 'normal',
        'parameters': {'mean': 0.0, 'std': 1.0}
    }

    tensor = utils.random_tensor(cfg, size=(1000,))

    assert tensor.shape == (1000,)
    # Check mean and std are approximately correct
    assert abs(tensor.mean().item() - 0.0) < 0.1
    assert abs(tensor.std().item() - 1.0) < 0.2


def test_random_tensor_normal_custom_params():
    """Test normal distribution with custom mean and std"""
    cfg = {
        'distribution': 'normal',
        'parameters': {'mean': 10.0, 'std': 2.0}
    }

    tensor = utils.random_tensor(cfg, size=(1000,))

    # Check mean is approximately 10
    assert abs(tensor.mean().item() - 10.0) < 0.5
    # Check std is approximately 2
    assert abs(tensor.std().item() - 2.0) < 0.5


def test_random_tensor_list_config():
    """Test list config with multiple distributions"""
    cfg = [
        {'distribution': 'uniform', 'parameters': {'lower': 0.0, 'upper': 1.0}},
        {'distribution': 'normal', 'parameters': {'mean': 5.0, 'std': 0.5}},
        {'distribution': 'uniform', 'parameters': {'lower': -1.0, 'upper': 1.0}}
    ]

    tensor = utils.random_tensor(cfg)

    assert len(tensor) == 3
    # First element should be in [0, 1]
    assert 0.0 <= tensor[0] <= 1.0
    # Second element should be around 5.0 (with some tolerance)
    assert 3.0 <= tensor[1] <= 7.0  # Allow wide range due to randomness
    # Third element should be in [-1, 1]
    assert -1.0 <= tensor[2] <= 1.0


def test_random_tensor_multidimensional():
    """Test multidimensional tensor generation"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 0.0, 'upper': 1.0}
    }

    for size in [(5,), (3, 4), (2, 3, 4)]:
        tensor = utils.random_tensor(cfg, size=size)
        assert tensor.shape == torch.Size(size)
        assert (tensor >= 0.0).all()
        assert (tensor <= 1.0).all()


def test_random_tensor_device_cpu():
    """Test tensor is created on specified device (CPU)"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 0.0, 'upper': 1.0}
    }

    tensor = utils.random_tensor(cfg, size=(10,), device='cpu')

    assert tensor.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_tensor_device_cuda():
    """Test tensor is created on CUDA device if available"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 0.0, 'upper': 1.0}
    }

    tensor = utils.random_tensor(cfg, size=(10,), device='cuda')

    assert tensor.device.type == 'cuda'


def test_random_tensor_invalid_distribution():
    """Test that invalid distribution raises ValueError"""
    cfg = {
        'distribution': 'invalid_dist',
        'parameters': {'param': 1.0}
    }

    with pytest.raises(ValueError, match="Unrecognised distribution type"):
        utils.random_tensor(cfg, size=(10,))


def test_random_tensor_extra_kwargs():
    """Test that extra kwargs are ignored"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 0.0, 'upper': 1.0}
    }

    # Should not raise error for extra kwargs
    tensor = utils.random_tensor(cfg, size=(10,), extra_param='ignored', another='also_ignored')

    assert tensor.shape == (10,)


def test_random_tensor_reproducibility():
    """Test that setting seed produces reproducible results"""
    cfg = {
        'distribution': 'uniform',
        'parameters': {'lower': 0.0, 'upper': 1.0}
    }

    torch.manual_seed(42)
    tensor1 = utils.random_tensor(cfg, size=(10,))

    torch.manual_seed(42)
    tensor2 = utils.random_tensor(cfg, size=(10,))

    assert torch.allclose(tensor1, tensor2)


def test_random_tensor_list_empty():
    """Test behaviour with empty list config"""
    cfg = []

    tensor = utils.random_tensor(cfg)

    assert len(tensor) == 0


def test_random_tensor_normal_zero_std():
    """Test normal distribution with zero std (all same value)"""
    cfg = {
        'distribution': 'normal',
        'parameters': {'mean': 5.0, 'std': 0.0}
    }

    tensor = utils.random_tensor(cfg, size=(10,))

    # All values should be exactly the mean
    assert torch.allclose(tensor, torch.tensor(5.0))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_workflow():
    """Test a complete workflow using all functions"""
    # Create a config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
                root_model_name: MyModel
                device: cpu
                num_threads: 1
                MyModel:
                  init_params:
                    distribution: uniform
                    parameters:
                      lower: 0.0
                      upper: 1.0
                  learning_rate: 0.01
                """)
        config_path = f.name

    try:
        # Load config
        cfg, model_cfg = utils.load_config(config_path)

        # Set device
        device = utils.set_default_device(cfg)

        # Generate random tensor
        init_tensor = utils.random_tensor(
            model_cfg['init_params'],
            size=(10, 10),
            device=str(device)
        )

        assert init_tensor.shape == (10, 10)
        assert (init_tensor >= 0.0).all()
        assert (init_tensor <= 1.0).all()
        assert init_tensor.device.type == 'cpu'
    finally:
        Path(config_path).unlink()


# ============================================================================
# EXISTING TEST (Keep for compatibility)
# ============================================================================

def test_random_tensor():
    """Original test from test file"""

    def _test_entry(cfg, tensor):
        if cfg["distribution"] == "uniform":
            assert cfg["parameters"]["lower"] <= tensor <= cfg["parameters"]["upper"]

    for _, config in test_cfg.items():
        _raises = config.pop("_raises", False)
        _exp_exc = Exception if not isinstance(_raises, str) else globals()[_raises]
        _warns = config.pop("_warns", False)
        _exp_warning = UserWarning if not isinstance(_warns, str) else globals()[_warns]
        _match = config.pop("_match", " ")

        cfg = config if "cfg" not in config.keys() else config.get("cfg")

        if not _raises:
            for size in [(1,), (4, 4, 4)]:
                t = utils.random_tensor(cfg, size=size)

                if isinstance(cfg, list):
                    assert len(t) == len(cfg)
                else:
                    assert t.shape == torch.Size(size)

                t = torch.flatten(t)
                for _ in range(len(t)):
                    if isinstance(cfg, dict):
                        _test_entry(cfg, t[_])
                    else:
                        _test_entry(cfg[_], t[_])

        if not _raises and not _warns:
            utils.random_tensor(cfg, size=(1,))

        elif _warns and not _raises:
            with pytest.warns(_exp_warning, match=_match):
                utils.random_tensor(cfg, size=(1,))

        elif _raises and not _warns:
            with pytest.raises(_exp_exc, match=_match):
                utils.random_tensor(cfg, size=(1,))