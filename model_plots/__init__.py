import logging

logging.basicConfig(format="%(levelname)-8s %(module)-14s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# Set matplotlib backend globally in order to avoid potential issues from
# people forgetting to set this
import matplotlib

matplotlib.use("Agg")

import model_plots.HarrisWilson

# --- Custom DAG operations to register --------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from utopya.eval import register_operation

register_operation(name="pd.Index", func=pd.Index)
register_operation(name="np.maximum", func=np.maximum)
register_operation(name="np.subtract", func=np.subtract)
register_operation(name="np.exp", func=np.exp)
register_operation(name=".idxmax", func=lambda d, *a, **k: d.idxmax(*a, **k))
register_operation(name=".stack", func=lambda d, *a, **k: d.stack(*a, **k))
from .data_ops import *




