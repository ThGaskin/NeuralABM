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
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

import model_plots.HarrisWilson
import model_plots.Kuramoto
from utopya.eval import register_operation

register_operation(name="pd.Index", func=pd.Index)
register_operation(name="np.maximum", func=np.maximum)
register_operation(name="np.subtract", func=np.subtract)
register_operation(name="np.exp", func=np.exp)
register_operation(name=".idxmax", func=lambda d, *a, **k: d.idxmax(*a, **k))
register_operation(name=".idxmin", func=lambda d, *a, **k: d.idxmin(*a, **k))
register_operation(name=".stack", func=lambda d, *a, **k: d.stack(*a, **k))
register_operation(name=".reindex", func=lambda d, *a, **k: d.reindex(*a, **k))
register_operation(name="np.sum", func=np.sum)
register_operation(name="nx.from_numpy_matrix", func=nx.from_numpy_matrix)
register_operation(name="nx.clustering", func=nx.clustering)
register_operation(name="np.nonzero", func=np.nonzero)
register_operation(name="xr.where", func=xr.where)
register_operation(name="xr.apply_ufunc", func=xr.apply_ufunc)
register_operation(name=".to_dataset", func=lambda d, *a, **k: d.to_dataset(*a, **k))
register_operation(name=".unstack", func=lambda d, *a, **k: d.unstack(*a, **k))
register_operation(name="sin", func=np.sin)
register_operation(name="cos", func=np.cos)
register_operation(name="zip", func=zip)
register_operation(name="np.around", func=np.around)
from .data_ops import *
