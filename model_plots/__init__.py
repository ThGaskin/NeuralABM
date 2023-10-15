import logging

logging.basicConfig(
    format="%(levelname)-8s %(module)-14s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

# Set matplotlib backend globally in order to avoid potential issues from
# people forgetting to set this
import matplotlib

matplotlib.use("Agg")

# --- Custom DAG operations to register --------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr

import model_plots.HarrisWilson
from utopya.eval import register_operation

register_operation(name="pd.Index", func=pd.Index)
register_operation(name="np.maximum", func=np.maximum)
register_operation(name="np.subtract", func=np.subtract)
register_operation(name="np.exp", func=np.exp)
register_operation(name="np.sin", func=np.sin)
register_operation(name="xr.where", func=xr.where)
register_operation(name=".idxmax", func=lambda d, *a, **k: d.idxmax(*a, **k))
register_operation(name=".idxmin", func=lambda d, *a, **k: d.idxmin(*a, **k))
register_operation(name=".stack", func=lambda d, *a, **k: d.stack(*a, **k))
register_operation(name=".to_xarray", func=lambda d, *a, **k: d.to_xarray(*a, **k))
register_operation(name="np.nansum", func=np.nansum)
register_operation(name="np.histogramdd", func=np.histogramdd)
from .data_ops import *
from .SIR_trajectories_from_densities import *
