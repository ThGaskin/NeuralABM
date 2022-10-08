import logging

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import scipy.ndimage
import xarray as xr

from utopya.eval import PlotHelper, is_plot_func

log = logging.getLogger(__name__)


@is_plot_func(use_dag=True)
def plot_prob_density(
    data: xr.Dataset,
    hlpr: PlotHelper,
    *,
    hue: str = None,
    info_box_labels: dict = None,
    smooth_kwargs: dict = {},
    **plot_kwargs
):

    """Plots the marginal probability densities for a collection of datasets consisting of *different*
    x-values (param1) and associated probability values ('prob'). If specified, smooths the densities using
    a Gaussian kernel."""

    # Get the 'data' !dag_tag
    dset = data.pop("data")

    # Get the smoothing properties
    smooth, sigma = smooth_kwargs.pop("enabled", False), smooth_kwargs.pop(
        "sigma", None
    )

    # Plot stacked lines
    if hue:
        for coord in dset.coords[hue].values:

            y_vals = dset["prob"].sel({hue: coord})

            # Smooth the probability distribution, if set
            if smooth:
                y_vals = scipy.ndimage.gaussian_filter1d(y_vals, sigma, **smooth_kwargs)

            # Plot the distribution
            hlpr.ax.plot(
                dset["param1"].sel({hue: coord}), y_vals, label=coord, **plot_kwargs
            )

    else:
        y_vals = dset["prob"]

        # Smooth the probability distribution, if set
        if smooth:
            y_vals = scipy.ndimage.gaussian_filter1d(y_vals, sigma, **smooth_kwargs)

        # Plot the distribution
        hlpr.ax.plot(dset["param1"], y_vals, **plot_kwargs)

    # Plot the textbox, if given, using the remaining !dag_tags, which should be floats
    if info_box_labels:
        legend = hlpr.ax.legend(
            [mlines.Line2D([], [], lw=0)] * len(data),
            [
                info_box_labels[key] + " = " + str(data[key])
                for key in list(data.keys())
            ],
            loc="best",
            handlelength=0,
            handleheight=0,
            handletextpad=0,
        )
        plt.gca().add_artist(legend)
