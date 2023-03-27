import logging

import xarray as xr

from utopya.eval import PlotHelper, is_plot_func

log = logging.getLogger(__name__)


@is_plot_func(
    use_dag=True,
    use_helper=True,
    required_dag_tags=("loss_data", "CPU_time", "GPU_time", "regression_time"),
)
def time_and_loss(
    hlpr: PlotHelper,
    *,
    data: dict,
    loss_color: str,
    CPU_color: str,
    GPU_color: str,
    regression_color: str
):
    """Plots a comparison of time and loss values onto two axes"""
    loss_data: xr.Dataset = data["loss_data"]
    CPU_time: xr.Dataset = data["CPU_time"]
    GPU_time: xr.Dataset = data["GPU_time"]
    regression_time: xr.Dataset = data["regression_time"]
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
    ax2.set_ylabel(r"avg. $L1$ error after 10 epochs")
    ax2.set_ylim([0.1, 0.5])

    hlpr.select_axis(ax=ax1)
    CPU_time["y"].plot(color=CPU_color)
    ax1.fill_between(
        CPU_time.coords["N"],
        (CPU_time["y"] + CPU_time["yerr"]),
        (CPU_time["y"] - CPU_time["yerr"]),
        alpha=0.5,
        color=CPU_color,
        lw=0,
    )

    GPU_time["y"].plot(color=GPU_color)
    ax1.fill_between(
        GPU_time.coords["N"],
        (GPU_time["y"] + GPU_time["yerr"]),
        (GPU_time["y"] - GPU_time["yerr"]),
        alpha=0.5,
        color=GPU_color,
        lw=0,
    )

    regression_time.plot(color=regression_color, ax=ax1, linestyle="dotted")
