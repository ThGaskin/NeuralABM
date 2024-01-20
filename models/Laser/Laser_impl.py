import coloredlogs
import logging
import numpy as np
import torch
from torch.fft import fft, ifft
from typing import Any

from torchdiffeq import odeint

# Set up the logger
log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s    %(message)s", level="INFO", logger=log)


# ----------------------------------------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------------------------------------


# Hyperbolic secant function
def sech(x) -> Any:
    return 1 / torch.cosh(x)


# ----------------------------------------------------------------------------------------------------------------------
# Numerical solver class for the CNLS
# ----------------------------------------------------------------------------------------------------------------------


class CNLS_Solver:
    def __init__(self, *, t: torch.Tensor, z: torch.Tensor, parameters: dict, **__):
        """Solver class for the coupled non-linear Schrödinger equations.

        :param t: time array
        :param z: space array
        :param parameters: dictionary of the cavity properties, containing:
            - D: chromatic dispersion; normal (D < 0) or anomalous (D > 0)
            - K: the birefringence of the optical fiber
            - E0: cavity saturation energy
            - Omega: bandwidth of the gain media (denoted by tau in original code) #TODO is this right?
            - g0: gain strength (pumping)
            - Gamma: distributed loss of the fiber laser cavity
            - A: non-linear coupling parameter, A + B = 1
            - B: non-linear coupling parameter of the fiber, A + B = 1.
        """

        # Time array
        self.t = t
        self.dt = t[1] - t[0]

        # Space array
        self.z = z
        self.dz = z[1] - z[0]

        # Frequency array, both in the natural order and sorted
        self.k = 2 * torch.pi / (t[-1] - t[0]) * torch.fft.fftfreq(len(t), 1/self.dt)
        self.k_sorted = torch.cat([self.k[int(len(self.k)/2):], self.k[:int(len(self.k)/2)]])

        # Cavity parameters
        self.parameters = parameters

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate the energy of the beam. The energy is given by

        .. math::
            \int u^2 + v^2 dt

        """
        u, v = state

        return torch.trapezoid(y=torch.conj(u) * u + torch.conj(v) * v, x=self.t).real

    def kurtosis(self, state: torch.Tensor) -> torch.Tensor:
        """ Calculate the kurtosis of a state, given by the sum of the kurtoses of each individual component.
        The kurtosis of a signal is given by the fourth moment about the mean divided by the square of the variance:

        .. math::
            E [ (u-\bar{u})^4 ] / E [ (u-\bar{u})^2 ] ^2

        """
        # Helper function for kurtosis along a single dimension
        def _kurtosis(_x):

            # Calculate the Fourier spectrum and normalise
            _spec = torch.abs(fft(_x))**2
            _spec = torch.cat([_spec[int(len(self.k)/2):], _spec[:int(len(self.k)/2)]])
            _norm = torch.trapz(y=_spec, x=self.k_sorted)
            _spec = _spec / _norm

            # Calculate the mean, standard deviation, and fourth moment
            _mean = torch.trapz(_spec * self.k_sorted, self.k_sorted)
            _m_2 = torch.trapz((self.k_sorted - _mean) ** 2 * _spec, self.k_sorted)
            _m_4 = torch.trapz((self.k_sorted - _mean) ** 4 * _spec, self.k_sorted)

            # Return the normalised fourth moment
            return _m_4 / (_m_2 ** 2)

        u, v = state
        return _kurtosis(u) + _kurtosis(v)

    def set_parameter(self, params: dict) -> None:
        """Update the dictionary of parameters"""
        self.parameters.update(params)

    def get_parameter(self, param: str) -> float:
        """Get the current value of a parameter"""
        return self.parameters[param]

    def solve(self, initial_conditions) -> np.ndarray:
        """Solve the coupled non-linear Schrödinger equations. The initial condition is given in real space,
        transformed into Fourier space in the time dimension, and then solved using a RK45 solver.

        :param initial_conditions: the initial condition of the solver, given in real space
        :return (u, v) at z = Z, i.e. at the end of the optical fiber, in real space
        """

        def _rhs_single(_x, _x_f, _y, _y_f, _E):
            """Helper function for the _cnls_rhs"""

            return (
                - torch.tensor(1j) * 0.5 * self.parameters["D"] * (self.k**2) * _x_f
                - torch.tensor(1j) * self.parameters["K"] * _x_f
                + torch.tensor(1j) * fft(
                    (torch.conj(_x) * _x + self.parameters["A"] * torch.conj(_y) * _y)
                    * _x
                    + self.parameters["B"] * (_y**2) * torch.conj(_x)
                )
                + 2 * self.parameters["g0"] / (1 + _E / self.parameters["E0"])
                * (1 - self.parameters["Omega"] * (self.k**2)) * _x_f
                - self.parameters["Gamma"] * _x_f
            )

        def _cnls_rhs(_z, _state):
            """Right-hand side of the CNLS. This function is passed to the solver. The state must be passed as a
            flattened array."""

            # u and v in real space
            _u, _v = _state[:int(len(_state) / 2)], _state[int(len(_state) / 2):]

            # u and v in Fourier space
            _u_f, _v_f = fft(_u), fft(_v)

            # Calculate the energy
            E = self.energy(torch.stack([_u, _v]))

            res = torch.stack(
                [
                    _rhs_single(_u, _u_f, _v, _v_f, E),
                    _rhs_single(_v, _v_f, _u, _u_f, E),
                ]
            )
            res = ifft(res, dim=1).flatten()

            return res

        # Use the torch.diffeq odeint solver with the Dopri5Solver (default)
        _solution = odeint(
            _cnls_rhs,
            t=self.z,
            y0=initial_conditions.flatten(),
            method='dopri5'
        )

        # Return the state of the beam after one round trip (final z-value)
        return _solution[-1, :].reshape(2, -1)


# ----------------------------------------------------------------------------------------------------------------------
# Laser cavity class
# ----------------------------------------------------------------------------------------------------------------------


class Laser_cavity:
    """The laser cavity class. The waveplate and polarizer matrices are class properties. The state of the laser is
    given by the current state of (u, v). The laser class contains member functions that allow for setting the waveplate
    angles, performing a single roundtrip and solving the CNLS, and resetting the laser.
    It can also calculate the energy of the beam at its current state."""

    # Quarter waveplate
    W4 = torch.tensor([[torch.exp(-torch.tensor(1j) * torch.pi / 4), 0], [0, torch.exp(torch.tensor(1j) * torch.pi / 4)]])
    # Half waveplate
    W2 = torch.tensor([[-torch.tensor(1j), 0], [0, torch.tensor(1j)]])
    # Polarizer
    WP = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat)

    # Stack the waveplates and polarizers into an array
    waveplates_and_polarizers = [W4, W4, W2, WP]

    def __init__(
        self,
        *,
        t: torch.Tensor,
        z: torch.Tensor,
        parameters: dict,
        alpha: torch.Tensor,
        initial_condition: torch.Tensor = None,
        **__
    ):
        """Initialises an instance of the laser.

        :param t: time array
        :param z: space array
        :param parameters: dictionary of the cavity properties, containing:
            - D: chromatic dispersion; normal (D < 0) or anomalous (D > 0)
            - K: the birefringence of the optical fiber
            - E0: cavity saturation energy
            - Omega: bandwidth of the gain media (denoted by tau in original code) #TODO is this right?
            - g0: gain strength (pumping)
            - Gamma: distributed loss of the fiber laser cavity
            - A: non-linear coupling parameter, A + B = 1
            - B: non-linear coupling parameter of the fiber, A + B = 1.
        :param alpha: initial angles of the waveplates
        :param initial_condition: (optional) an initial condition for (u, v), given in real space

        """

        # Initialise the numerical solver
        self.solver = CNLS_Solver(t=t, z=z, parameters=parameters)

        # Angles of the polarizers
        self._alpha = alpha

        # Transformation matrix as a function of alpha
        self._transformation_matrix = self.calculate_transformation_matrix()

        # Store the initial state of the laser, which can be used for resetting
        self._init_state = initial_condition
        self._init_alpha = alpha

        # Current state of u and v
        self.state = initial_condition

        # Current energy
        self.energy = None

        # Current kurtosis
        self.kurtosis = None

        # Current time
        self._t = 0

    def calculate_transformation_matrix(self, alpha: torch.Tensor = None) -> torch.Tensor:
        """Gets the current state of the laser cavity"""

        alpha = self._alpha if alpha is None else alpha

        # Calculate the rotation matrices from the current angle
        rotation_matrices = [
            torch.stack(
                [torch.cos(_alpha), -torch.sin(_alpha), torch.sin(_alpha), torch.cos(_alpha)]
            ).reshape(2, -1).cfloat()
            for _alpha in alpha
        ]

        # Return the transformation matrix: J_1 * J_p * J_2 * J_3
        J = [
            rotation_matrices[i]
            @ self.waveplates_and_polarizers[i]
            @ rotation_matrices[i].T
            for i in range(4)
        ]
        return J[0] @ J[-1] @ J[1] @ J[2]

    def transformation_matrix(self):
        """ Gets the transformation matrix """
        return self._transformation_matrix

    def set_transformation_matrix(self, transformation_matrix: torch.Tensor) -> None:
        """ Sets the transformation matrix"""
        self._transformation_matrix = transformation_matrix

    def apply_transformation(self, state) -> torch.Tensor:
        """Applies the transformation of the waveplates to a given state"""
        return self._transformation_matrix @ state

    def set_alpha(self, alpha: torch.Tensor) -> None:
        """Setter function for alpha. When changing the angles of the waveplates and polarizers, the
        transformation matrix is automatically adjusted."""
        self._alpha = alpha
        self._transformation_matrix = self.calculate_transformation_matrix(alpha)

    def set_state(self, state: torch.Tensor, *, set_init: bool = False) -> None:
        """Setter function for the current state of (u, v). If specified, also sets this state as the initial
        condition"""
        self.state = state
        if set_init:
            self._init_state = self.state

    def set_energy(self):
        """Calculates and sets the energy of the current state"""
        self.energy = self.solver.energy(self.state)

    def alpha(self):
        """ Returns the current angles of the waveplates """
        return self._alpha

    def reset(self) -> None:
        """Resets the laser to its initial state"""
        self._alpha = self._init_alpha
        self.state = self._init_state
        self._t = 0

    def clear_gradients(self) -> None:
        """ Clears the gradients of the parameters"""
        self._alpha = self._alpha.detach()
        self.state = self.state.detach()

    def round_trip(self, *, update_self: bool = True) -> tuple:
        """Performs a round trip of the beam around the laser cavity, using the solver, and applies the
         waveplate transformation. The transformation is then applied, and the state after
         the transformation returned, as well the updated energy. The state of the cavity is then updated to
        in preparation for a new round trip (optional).

        :param update_self: (optional) whether to update the state of the laser. True by default, but can be turned off
            for testing.
        :returns: tuple of the new state behind the waveplates and the new energy
        """

        # Solve the PDE with the current state as the initial condition
        new_state = self.solver.solve(self.state)

        # Calculate the energy of the new state
        new_energy = self.solver.energy(new_state)

        # Apply the waveplate transformation to the current state
        new_state = self.apply_transformation(new_state)

        if update_self:
            # Apply the transformation matrix
            self.state = new_state

            # Update the energy
            self.energy = new_energy

            # Update the kurtosis
            self.kurtosis = self.solver.kurtosis(new_state)

            self._t += 1

        return new_state, new_energy
