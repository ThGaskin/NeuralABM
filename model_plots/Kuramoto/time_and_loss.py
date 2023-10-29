import logging

import xarray as xr

from utopya.eval import PlotHelper, is_plot_func

log = logging.getLogger(__name__)


@is_plot_func(
    use_dag=True,
    use_helper=True,
    required_dag_tags=("loss_data", "neural_time", "MCMC_time"),
)
def time_and_loss(
    hlpr: PlotHelper,
    *,
    data: dict,
    loss_color: str,
    neural_color: str = None,
    MCMC_color: str = None,
):
    """Plots a comparison of time and loss values onto two axes"""
    loss_data: xr.Dataset = data["loss_data"]
    neural_time: xr.Dataset = data["neural_time"]
    MCMC_time: xr.Dataset = data["MCMC_time"]
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
    ax2.set_ylabel(r"avg. $L^1$ error after 10 epochs")
    ax2.set_ylim([0.1, 0.5])

    # Plot the compute times
    hlpr.select_axis(ax=ax1)
    neural_time["y"].plot(color=neural_color)
    ax1.fill_between(
        neural_time.coords["N"],
        (neural_time["y"] + neural_time["yerr"]),
        (neural_time["y"] - neural_time["yerr"]),
        alpha=0.5,
        color=neural_color,
        lw=0,
    )

    MCMC_time["y"].plot(color=MCMC_color)
    ax1.fill_between(
        MCMC_time.coords["N"],
        (MCMC_time["y"] + MCMC_time["yerr"]),
        (MCMC_time["y"] - MCMC_time["yerr"]),
        alpha=0.5,
        color=MCMC_color,
        lw=0,
    )
