import pytest
import torch
import math
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path

# Import the module to test
sys.path.insert(0, up(up(up(__file__))))

include = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="include"
)
from include.solvers import build_time_grid, torchdiffeq_solver  # Replace 'your_module' with actual module name


# ============================================================================
# build_time_grid TESTS
# ============================================================================

def test_build_time_grid_with_t():
    """Test build_time_grid when t is provided directly"""
    t = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    result = build_time_grid(t=t)

    assert torch.allclose(result, t)
    assert result.dtype == t.dtype


def test_build_time_grid_with_t_span_and_dt():
    """Test build_time_grid with t_span and dt"""
    t_span = (0.0, 1.0)
    dt = 0.1
    result = build_time_grid(t_span=t_span, dt=dt)

    expected = torch.arange(11) * 0.1
    assert torch.allclose(result, expected, atol=1e-7)
    assert len(result) == 11  # 0.0, 0.1, ..., 1.0


def test_build_time_grid_non_zero_start():
    """Test build_time_grid with non-zero start time"""
    t_span = (1.0, 2.0)
    dt = 0.2
    result = build_time_grid(t_span=t_span, dt=dt)

    expected = torch.tensor([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    assert torch.allclose(result, expected, atol=1e-7)


def test_build_time_grid_uneven_division():
    """Test build_time_grid when (t1-t0)/dt is not an integer"""
    t_span = (0.0, 1.0)
    dt = 0.3
    result = build_time_grid(t_span=t_span, dt=dt)

    # Should have ceil((1.0-0.0)/0.3) + 1 = 4 + 1 = 5 points
    assert len(result) == 5
    assert result[0] == 0.0
    assert result[-1] == 1.2  # Goes slightly beyond t1


def test_build_time_grid_device_and_dtype():
    """Test build_time_grid respects device and dtype"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dtype = torch.float64

    # Test with t provided
    t = torch.tensor([0.0, 1.0, 2.0])
    result = build_time_grid(t=t, device=device, dtype=dtype)
    assert result.device == device
    assert result.dtype == dtype

    # Test with t_span and dt
    result = build_time_grid(t_span=(0.0, 1.0), dt=0.1, device=device, dtype=dtype)
    assert result.device == device
    assert result.dtype == dtype


def test_build_time_grid_missing_parameters():
    """Test build_time_grid raises error when required parameters are missing"""
    # No parameters at all
    with pytest.raises(ValueError, match="Provide either"):
        build_time_grid()

    # Only t_span without dt
    with pytest.raises(ValueError, match="Provide either"):
        build_time_grid(t_span=(0.0, 1.0))

    # Only dt without t_span
    with pytest.raises(ValueError, match="Provide either"):
        build_time_grid(dt=0.1)


def test_build_time_grid_backward_integration():
    """Test build_time_grid with backward time integration"""
    t_span = (1.0, 0.0)
    dt = -0.1
    result = build_time_grid(t_span=t_span, dt=dt)

    expected = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert torch.allclose(result, expected, atol=1e-7)


def test_build_time_grid_list_input():
    """Test build_time_grid accepts list as t"""
    t_list = [0.0, 0.5, 1.0]
    result = build_time_grid(t=t_list)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(result, expected)


# ============================================================================
# torchdiffeq_solver DECORATOR TESTS
# ============================================================================

def test_simple_ode_exponential_growth():
    """Test solver on simple ODE: dy/dt = y, solution is y(t) = y0 * exp(t)"""

    @torchdiffeq_solver(method="dopri5")
    def exponential_rhs(t, y):
        return y

    y0 = torch.tensor([1.0])
    t, y = exponential_rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1)

    # Analytical solution: y(1) = e^1 ≈ 2.718
    expected_final = math.e
    assert torch.allclose(y[-1], torch.tensor([expected_final]), rtol=1e-4)
    assert len(t) == len(y)
    assert t[0] == 0.0
    assert torch.allclose(t[-1], torch.tensor(1.0))


def test_simple_ode_with_custom_time_grid():
    """Test solver with custom time grid"""

    @torchdiffeq_solver(method="dopri5")
    def exponential_rhs(t, y):
        return y

    y0 = torch.tensor([1.0])
    t_custom = torch.tensor([0.0, 0.5, 1.0, 2.0])
    t, y = exponential_rhs(y0=y0, t=t_custom)

    assert torch.allclose(t, t_custom)
    assert len(y) == len(t_custom)
    # Check y(2) = e^2
    assert torch.allclose(y[-1], torch.tensor([math.e ** 2]), rtol=1e-4)


def test_ode_with_parameters():
    """Test solver with parameterized ODE: dy/dt = k*y"""

    @torchdiffeq_solver(method="dopri5")
    def parameterized_rhs(t, y, k):
        return k * y

    k = 2.0
    y0 = torch.tensor([1.0])
    t, y = parameterized_rhs(k, y0=y0, t_span=(0.0, 1.0), dt=0.1)

    # Solution: y(t) = exp(k*t)
    expected_final = math.exp(k * 1.0)
    assert torch.allclose(y[-1], torch.tensor([expected_final]), rtol=1e-4)


def test_ode_with_kwargs():
    """Test solver with keyword arguments in RHS"""

    @torchdiffeq_solver(method="dopri5")
    def rhs_with_kwargs(t, y, scale=1.0, offset=0.0):
        return scale * y + offset

    y0 = torch.tensor([1.0])
    t, y = rhs_with_kwargs(y0=y0, t_span=(0.0, 0.5), dt=0.1, scale=2.0, offset=0.0)

    # With scale=2, offset=0: dy/dt = 2y, solution is y(t) = exp(2t)
    expected_final = math.exp(2.0 * 0.5)
    assert torch.allclose(y[-1], torch.tensor([expected_final]), rtol=1e-4)

def test_solver_with_different_methods():
    """Test that different solver methods work"""
    methods = ['euler', 'rk4', 'dopri5']

    @torchdiffeq_solver(method='dopri5')  # Default, will be overridden
    def simple_rhs(t, y):
        return y

    y0 = torch.tensor([1.0])

    for method in methods:
        @torchdiffeq_solver(method=method)
        def rhs(t, y):
            return y

        t, y = rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1)

        # All methods should give reasonable results (within order of magnitude)
        expected = math.e
        assert 0.5 * expected < y[-1].item() < 1.5 * expected, f"Method {method} failed"


def test_solver_with_adjoint():
    """Test solver with adjoint method"""

    @torchdiffeq_solver(method="dopri5", adjoint=True, adjoint_params=())
    def exponential_rhs(t, y):
        return y

    y0 = torch.tensor([1.0], requires_grad=True)
    t, y = exponential_rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1)

    # Test that we can backpropagate
    loss = y[-1].sum()
    loss.backward()

    assert y0.grad is not None
    assert not torch.isnan(y0.grad).any()


def test_solver_tolerances():
    """Test that different tolerances affect accuracy"""

    @torchdiffeq_solver(method="dopri5", rtol=1e-3, atol=1e-6)
    def rhs_low_tol(t, y):
        return y

    @torchdiffeq_solver(method="dopri5", rtol=1e-9, atol=1e-12)
    def rhs_high_tol(t, y):
        return y

    y0 = torch.tensor([1.0])
    t_custom = torch.tensor([0.0, 1.0])

    _, y_low = rhs_low_tol(y0=y0, t=t_custom)
    _, y_high = rhs_high_tol(y0=y0, t=t_custom)

    # High tolerance should be more accurate
    expected = torch.tensor([math.e])
    error_low = torch.abs(y_low[-1] - expected)
    error_high = torch.abs(y_high[-1] - expected)

    assert error_high < error_low or error_high < 1e-8


def test_solver_device_dtype_propagation():
    """Test that device and dtype are properly propagated"""

    @torchdiffeq_solver(method="dopri5")
    def simple_rhs(t, y):
        return y

    # Test with float64
    y0 = torch.tensor([1.0], dtype=torch.float64)
    t, y = simple_rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1)

    assert y.dtype == torch.float64
    assert t.dtype == torch.float64

    # Test with explicit device/dtype
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    y0 = torch.tensor([1.0])
    t, y = simple_rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1,
                      device=device, dtype=torch.float32)

    assert y.device == device
    assert y.dtype == torch.float32


def test_solver_output_shapes():
    """Test that output shapes are correct"""

    @torchdiffeq_solver(method="dopri5")
    def multidim_rhs(t, y):
        return torch.zeros_like(y)

    # Test 1D initial condition
    y0 = torch.randn(5)
    t, y = multidim_rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1)
    assert y.shape == (11, 5)  # (n_timesteps, y0_dim)

    # Test 2D initial condition
    y0 = torch.randn(3, 4)
    t, y = multidim_rhs(y0=y0, t_span=(0.0, 1.0), dt=0.1)
    assert y.shape == (11, 3, 4)  # (n_timesteps, *y0.shape)


def test_solver_preserves_function_metadata():
    """Test that decorator preserves function name and docstring"""

    @torchdiffeq_solver(method="dopri5")
    def my_special_ode(t, y, param):
        """This is a special ODE."""
        return param * y

    assert my_special_ode.__name__ == "my_special_ode"
    assert "special ODE" in my_special_ode.__doc__


def test_lorenz_system():
    """Test on the chaotic Lorenz system"""

    @torchdiffeq_solver(method="dopri5", rtol=1e-6, atol=1e-9)
    def lorenz(t, y, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        x, y_coord, z = y
        dx = sigma * (y_coord - x)
        dy = x * (rho - z) - y_coord
        dz = x * y_coord - beta * z
        return torch.stack([dx, dy, dz])

    y0 = torch.tensor([1.0, 1.0, 1.0])
    t, y = lorenz(y0=y0, t_span=(0.0, 10.0), dt=0.01)

    # Basic sanity checks
    assert not torch.isnan(y).any(), "Solution contains NaN"
    assert not torch.isinf(y).any(), "Solution contains Inf"
    assert y.shape[0] == len(t)
    assert y.shape[1] == 3


def test_stiff_ode():
    """Test on a stiff ODE (van der Pol oscillator)"""

    @torchdiffeq_solver(method="dopri5", rtol=1e-6, atol=1e-9)
    def van_der_pol(t, y, mu=1000.0):
        x, v = y
        dx = v
        dv = mu * (1 - x ** 2) * v - x
        return torch.stack([dx, dv])

    y0 = torch.tensor([2.0, 0.0])
    t, y = van_der_pol(y0=y0, t_span=(0.0, 0.1), dt=0.01, mu=10.0)

    # Should complete without errors
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_solver_with_time_dependent_forcing():
    """Test ODE with explicit time dependence"""

    @torchdiffeq_solver(method="dopri5")
    def forced_oscillator(t, y, omega=1.0):
        x, v = y
        # Forcing term: sin(omega * t)
        force = torch.sin(omega * t)
        dx = v
        dv = -x + force
        return torch.stack([dx, dv])

    y0 = torch.tensor([0.0, 0.0])
    t, y = forced_oscillator(y0=y0, t_span=(0.0, 10.0), dt=0.1, omega=2.0)

    assert not torch.isnan(y).any()
    assert y.shape == (101, 2)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

def test_zero_timestep():
    """Test with dt=0 should work for single point"""
    t = torch.tensor([0.0])
    result = build_time_grid(t=t)
    assert len(result) == 1


def test_very_small_timestep():
    """Test with very small timestep"""
    t_span = (0.0, 0.001)
    dt = 0.0001
    result = build_time_grid(t_span=t_span, dt=dt)
    assert len(result) == 11  # 0 to 0.001 with step 0.0001


def test_negative_time_span():
    """Test backward integration with negative dt"""

    @torchdiffeq_solver(method="dopri5")
    def simple_rhs(t, y):
        return y

    y0 = torch.tensor([1.0])
    t, y = simple_rhs(y0=y0, t_span=(1.0, 0.0), dt=-0.1)

    assert t[0] == 1.0
    assert t[-1] == 0.0
    assert torch.all(t[:-1] > t[1:])  # Decreasing