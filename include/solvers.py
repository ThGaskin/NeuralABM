import math
import torch
from functools import wraps
from typing import Callable, Optional, Tuple, Dict
from torchdiffeq import odeint, odeint_adjoint

Tensor = torch.Tensor

def build_time_grid(t: Tensor = None, t_span: tuple = None, dt: float = None, *,
                    device = None, dtype = None) -> Tensor:

    # Build time grid if not supplied
    if t is None:
        if t_span is None or dt is None:
            raise ValueError("Provide either `t` or (`t_span` and `dt`).")
        t0, t1 = t_span
        n_steps = int(math.ceil((t1 - t0) / dt))
        return torch.arange(n_steps + 1, device=device, dtype=dtype) * dt + t0
    else:
        return torch.as_tensor(t, device=device, dtype=dtype)

# ---------------------------------------------------------------------
# Decorator factory over torchdiffeq
# ---------------------------------------------------------------------
def torchdiffeq_solver(method: str = "dopri5",
                       adjoint: bool = False,
                       rtol: float = 1e-6,
                       atol: float = 1e-9,
                       options: Optional[Dict] = None):
    """
    Turns an RHS f(t, y, *args, **kwargs) into a solver that calls torchdiffeq.
    The decorated function signature is:
        solve(*args, y0, t=None, t_span=None, dt=None, device=None, dtype=None, **kwargs)
    You must provide either `t` OR (`t_span` and `dt`).
    """
    integrator = odeint_adjoint if adjoint else odeint
    options = {} if options is None else options

    def decorator(rhs: Callable):
        @wraps(rhs)
        def solve(*args,
                  y0,
                  t: Optional[Tensor] = None,
                  t_span: Optional[Tuple[float, float]] = None,
                  dt: Optional[float] = None,
                  device: Optional[torch.device] = None,
                  dtype: Optional[torch.dtype] = None,
                  **kwargs):
            # Prepare initial condition
            y0 = torch.as_tensor(y0, device=device, dtype=dtype)
            device, dtype = y0.device, y0.dtype

            # Build time grid
            t = build_time_grid(t, t_span, dt, device=device, dtype=dtype)

            # Closure that captures args/kwargs
            def fun(ti, yi):
                return rhs(ti, yi, *args, **kwargs)

            y = integrator(fun, y0, t, rtol=rtol, atol=atol, method=method, options=options)
            # y shape: (len(t), *y0.shape)
            return t, y
        return solve
    return decorator