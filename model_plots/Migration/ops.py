from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from utopya.eval import is_operation


@is_operation("Migration.save_flow_table_stats")
def save_bilateral_flow_stats(flow_samples, prob, path) -> None:

    prob_stacked = np.reshape(prob.data, (len(prob.coords["seed"]), len(prob.coords["batch"]), 1, 1, 1, 1))
    prob_stacked = np.repeat(prob_stacked, len(flow_samples.coords["Year"]), 2)
    prob_stacked = np.repeat(prob_stacked, len(flow_samples.coords["Direction"]), 3)
    prob_stacked = np.repeat(prob_stacked, len(flow_samples.coords["Origin ISO"]), 4)
    prob_stacked = np.repeat(prob_stacked, len(flow_samples.coords["Destination ISO"]), 5)

    mean = (flow_samples * prob_stacked).sum(["seed", "batch"])
    std = np.sqrt((((flow_samples - mean)**2) * prob_stacked).sum(["seed", "batch"]))
    # mean = flow_samples.mean(["seed", "batch"])
    # std = flow_samples.std(["seed", "batch"])
    stats_xr = xr.Dataset(dict(mean=mean, std=std))
    stats_xr.to_netcdf(path)