import copy
import logging

import scipy.ndimage
import xarray as xr
from dantro.plot.funcs._utils import plot_errorbar
from dantro.plot.funcs.generic import make_facet_grid_plot

from utopya.eval import PlotHelper, is_plot_func


@make_facet_grid_plot(
    map_as="dataset",
    encodings=("x", "y", "yerr", "hue", "col", "row", "alpha", "lw"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
)
def plot_prob_density(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    _is_facetgrid: bool,
    x: str,
    y: str,
    yerr: str = None,
    hue: str = None,
    label: str = None,
    add_legend: bool = True,
    smooth_kwargs: dict = {},
    **plot_kwargs,
):
    """Probability density plot for estimated parameters, which combines line- and errorband functionality into a
    single plot. Crucially, the x-value does not need to be a dataset coordinate. Is xarray facet_grid compatible.

    :param ds: dataset to plot
    :param hlpr: PlotHelper
    :param _is_facetgrid: whether the plot is a facet_grid instance or not (determined by the decorator function)
    :param x: coordinate or variable to use as the x-value
    :param y: values to plot onto the y-axis
    :param yerr (optional): variable to use for the errorbands. If None, no errorbands are plotted.
    :param hue: (optional) variable to plot onto the hue dimension
    :param label: (optional) label for the plot, if the hue dimension is unused
    :param add_legend: whether to add a legend
    :param smooth_kwargs: dictionary for the smoothing settings. Smoothing can be set for all parameters or by parameter
    :param plot_kwargs: passed to the plot function
    """

    def _plot_1d(*, _x, _y, _yerr, _smooth_kwargs, _ax, _label=None, **_plot_kwargs):

        """Plots a single parameter density and smooths the marginal. Returns the artists for the legend."""
        smooth, sigma = _smooth_kwargs.pop("enabled", False), _smooth_kwargs.pop(
            "sigma", None
        )
        # Smooth the y values, if given
        if smooth:
            _y = scipy.ndimage.gaussian_filter1d(_y, sigma, **_smooth_kwargs)

        # If no yerr is given, plot a single line
        if _yerr is None:
            (ebar,) = hlpr.ax.plot(_x, _y, label=_label, **_plot_kwargs)
            return ebar

        # Else, plot errorbands
        else:
            # Smooth the y error, if set
            if smooth:
                _yerr = scipy.ndimage.gaussian_filter1d(_yerr, sigma, **_smooth_kwargs)

            return plot_errorbar(
                ax=_ax,
                x=_x,
                y=_y,
                yerr=_yerr,
                label=_label,
                fill_between=True,
                **_plot_kwargs,
            )

    # Get the dataset and parameter name
    dset = ds
    if "parameter" in list(ds.coords):
        pname = ds.coords["parameter"].values.item()
    else:
        pname = list(ds.coords.keys())[0]

    # Track the legend handles and labels
    _handles, _labels = [], []
    if hue:
        for i, coord in enumerate(dset.coords[hue].values):

            if x in dset.coords:
                x_vals = dset.coords[x]
            else:
                x_vals = dset[x].sel({hue: coord})

            y_vals = dset[y].sel({hue: coord})
            yerr_vals = dset[yerr].sel({hue: coord}) if yerr is not None else None

            handle = _plot_1d(
                _x=x_vals,
                _y=y_vals,
                _yerr=yerr_vals,
                _smooth_kwargs=copy.deepcopy(smooth_kwargs.get(pname, smooth_kwargs)),
                _ax=hlpr.ax,
                _label=f"{coord}",
                **plot_kwargs,
            )

            _handles.append(handle)
            _labels.append(f"{coord}")

        if not _is_facetgrid:
            if add_legend:
                hlpr.ax.legend(_handles, _labels, title=hue)
        else:
            hlpr.track_handles_labels(_handles, _labels)
            if add_legend:
                hlpr.provide_defaults("set_figlegend", title=hue)

    else:

        if x in dset.coords:
            x_vals = dset.coords[x]
        else:
            x_vals = dset[x]
        y_vals = dset[y]
        yerr_vals = dset[yerr] if yerr is not None else None

        _plot_1d(
            _x=x_vals,
            _y=y_vals,
            _yerr=yerr_vals,
            _ax=hlpr.ax,
            _smooth_kwargs=copy.deepcopy(smooth_kwargs.get(pname, smooth_kwargs)),
            _label=label,
            **plot_kwargs,
        )


# ----------------------------------------------------------------------------------------------------------------------
# HACKY STUFF
# ----------------------------------------------------------------------------------------------------------------------
from utopya.eval import PlotHelper


def add_entry_to_figlegend(color: str, label: str, *, hlpr: PlotHelper):
    """Hacky solution.
    Adds an entry to an existing legend by copying the existing artists, re-drawing them in the given colour,
    and adding a label."""

    from dantro.plot.utils.mpl import remove_duplicate_handles_labels

    # Get the handles and labels from the current legend
    h, l = remove_duplicate_handles_labels(*hlpr.all_handles_labels)
    new_h = copy.deepcopy(h[0])
    new_h.set_color(color)
    h.append(new_h)
    l.append(label)

    title = hlpr._figlegend.get_title().get_text()
    hlpr._figlegend.remove()

    # Get the title and position of current legend
    hlpr.fig.legend(h, l, title=title, loc="center right")


def calculate_Hellingers(*, hlpr: PlotHelper):

    import matplotlib
    import numpy as np
    import scipy

    log = logging.getLogger(__name__)

    objs = dict()
    plots = [
        obj
        for obj in (hlpr.ax.get_children())
        if isinstance(obj, matplotlib.lines.Line2D)
    ]

    # Get the data
    for line in plots:
        if line.get_color() == "#2F7194":
            objs["Neural"] = line.get_data()
        elif line.get_color() == "#3D4244":
            objs["True"] = line.get_data()
        else:
            objs["MCMC"] = line.get_data()

    if "True" not in objs.keys():
        objs["True"] = [np.linspace(0, 1, 1000), np.ones(1000)]
    Hellingers = dict()
    # Interpolate: get the highest common lower bound, and lowest common upper bound
    for val in ["Neural", "MCMC"]:
        x_min, x_max = np.max([np.min(objs["True"][0]), np.min(objs[val][0])]), np.min(
            [np.max(objs["True"][0]), np.max(objs[val][0])]
        )
        grid = np.linspace(x_min, x_max, 1000)
        true_interp = np.interp(grid, objs["True"][0], objs["True"][1])
        val_interp = np.interp(grid, objs[val][0], objs[val][1])
        Hellinger = 0.5 * scipy.integrate.trapezoid(
            pow(np.sqrt(val_interp) - np.sqrt(true_interp), 2), grid
        )
        Hellingers[val] = Hellinger
    log.remark(f"Hellinger distances: {Hellingers}")
