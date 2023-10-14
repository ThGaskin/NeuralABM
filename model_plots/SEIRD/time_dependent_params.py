import numpy as np
import xarray as xr

from utopya.eval import is_operation


@is_operation("SEIRD+_time_dependent_parameters")
def time_dependent_parameters(cfg) -> xr.DataArray:

    """Returns an xr.Dataset of all the time-dependent parameters specified in the config over time"""

    # Load the time-dependent parameters
    time_dependent_params = cfg.get("time_dependent_params", {})

    # Get the number of steps
    num_steps = cfg["synthetic_data"]["num_steps"]

    res = []
    for parameter, intervals in time_dependent_params.items():
        i = 0
        val = np.zeros(num_steps)
        for interval in intervals:
            if not interval[-1]:
                interval[-1] = num_steps
            for t in np.arange(*interval):
                val[t] = cfg["synthetic_data"][parameter][i]
            i += 1
        res.append(
            xr.DataArray(
                data=[val],
                dims=["parameter", "time"],
                coords=dict(parameter=[parameter], time=np.arange(num_steps)),
            )
        )

    return xr.concat(res, dim="parameter")
