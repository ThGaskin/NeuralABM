import logging
from typing import Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import scipy.ndimage
import xarray as xr
from dantro.plot.funcs.generic import errorbars

from utopya.eval import PlotHelper, is_plot_func

log = logging.getLogger(__name__)


@is_plot_func(
    use_dag=True,
    use_helper=True,
    required_dag_tags=(
        "loss_data",
        "time_data",
    ),
)
def time_and_loss(hlpr: PlotHelper, *, data: dict, loss_color: str, time_color: str):
    loss_data: xr.Dataset = data["loss_data"]
    time_data: xr.Dataset = data["time_data"]

    ax1 = hlpr.ax

    # Plot the loss data
    ax2 = hlpr.ax.twinx()
    loss_data["y"].plot(ax=ax2, color=loss_color)
    ax2.fill_between(
        loss_data.coords["N"],
        (loss_data["y"] + loss_data["yerr"]),
        (loss_data["y"] - loss_data["yerr"]),
        alpha=0.2,
        color=loss_color,
        lw=0,
    )
    ax2.set_ylabel(r"$L1$ error after 10 epochs")
    ax2.set_ylim([0.1, 0.5])

    hlpr.select_axis(ax=ax1)
    time_data["y"].plot(color=time_color)
    ax1.fill_between(
        time_data.coords["N"],
        (time_data["y"] + time_data["yerr"]),
        (time_data["y"] - time_data["yerr"]),
        alpha=0.5,
        color=time_color,
        lw=0,
    )
