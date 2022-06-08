import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.ndimage
import xarray as xr
import numpy as np
from typing import Union
import matplotlib.colors as mcolors
from utopya.plotting import is_plot_func, PlotHelper
from dantro.plot.utils import ColorManager
from matplotlib.colors import LightSource
log = logging.getLogger(__name__)


@is_plot_func(use_dag=True)
def plot_marginals_3d(data: xr.Dataset,
                      hlpr: PlotHelper,
                      *,
                      cmap: Union[str, dict, mcolors.Colormap] = None,
                      norm: Union[str, dict, mcolors.Normalize] = None,
                      vmin: float = None,
                      vmax: float = None,
                      marginals_a: dict = None,
                      marginals_b: dict = None,
                      smooth_kwargs: dict = None,
                      add_colorbar: bool = True,
                      cbar_kwargs: dict = None,
                      **kwargs,):
    cm = ColorManager(
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    dset = data.pop('data')

    dset.plot.surface(x='x', y='y', ax=hlpr.ax, cmap=cm.cmap, vmin=cm.vmin, vmax=cm.vmax,
                      rstride=1, cstride=1)



    marginals_x = data.pop('marginals_a')
    marginals_y = data.pop('marginals_b')
    sigma = 0.5
    hlpr.ax.plot(marginals_y['param1'], np.zeros_like(marginals_y['param1']),
                 scipy.ndimage.gaussian_filter1d(marginals_y['prob'], sigma, ),
                 **marginals_a)
    hlpr.ax.plot(np.zeros_like(marginals_x['param1']), marginals_x['param1'],
                 scipy.ndimage.gaussian_filter1d(marginals_x['prob'], sigma, ),
                 **marginals_b)




