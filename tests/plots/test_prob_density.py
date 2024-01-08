import sys
from os.path import dirname as up

import numpy as np
import xarray as xr
from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

plot = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="model_plots.prob_density"
)
