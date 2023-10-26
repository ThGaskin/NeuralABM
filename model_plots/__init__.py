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
register_operation(name="np.random.randint", func=np.random.randint)
register_operation(name="xr.where", func=xr.where)
register_operation(name=".idxmax", func=lambda d, *a, **k: d.idxmax(*a, **k))
register_operation(name=".idxmin", func=lambda d, *a, **k: d.idxmin(*a, **k))
register_operation(name=".stack", func=lambda d, *a, **k: d.stack(*a, **k))
register_operation(name=".to_xarray", func=lambda d, *a, **k: d.to_xarray(*a, **k))
register_operation(
    name="pd.to_datetime", func=lambda d, *a, **k: pd.to_datetime(d, *a, **k)
)
register_operation(name="pd.date_range", func=lambda *a, **k: pd.date_range(*a, **k))
register_operation(name=".index", func=lambda d: d.index)
register_operation(name=".reset_index", func=lambda d, *a, **k: d.reset_index(*a, **k))
register_operation(name="replace", func=lambda s, *a, **k: s.replace(*a, **k))
register_operation(name="np.nansum", func=np.nansum)
register_operation(name="np.histogramdd", func=np.histogramdd)
register_operation(name="replace", func=lambda s, *a, **k: s.replace(*a, **k))
register_operation(name="list_of", func=lambda s: [s])
register_operation(name="zip", func=zip)
register_operation(name=".capitalize", func=lambda s: s.capitalize())
register_operation(name="np.randint", func=np.random.randint)

from model_plots.SEIRD.time_dependent_params import *

register_operation(name=".to_xarray", func=lambda d, *a, **k: d.to_xarray(*a, **k))
register_operation(name="np.nansum", func=np.nansum)
register_operation(name="np.histogramdd", func=np.histogramdd)
register_operation(name="np.ones", func=np.ones)
from .data_ops import *
from .nw_ops import *
from .SEIRD_trajectories_from_densities import SEIRD_densities_from_joint
from .SIR_trajectories_from_densities import *
