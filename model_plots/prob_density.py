import copy
from typing import Sequence, Union

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
    register_as_kind='density'
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
    linestyle: Union[str, Sequence] = "solid",
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
            "smoothing", None
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
    if "parameter" in list(ds.coords):
        pname = ds.coords["parameter"].values.item()
    else:
        for _c in ds.coords:
            # Exclude 1D variables and the hue variable
            if ds.coords[_c].shape == ():
                continue
            if hue is not None and _c == hue:
                continue
            pname = _c

    # Track the legend handles and labels
    _handles, _labels = [], []
    if hue:
        for i, coord in enumerate(ds.coords[hue].values):
            if x in ds.coords:
                x_vals = ds.coords[x]
            else:
                x_vals = ds[x].sel({hue: coord})

            y_vals = ds[y].sel({hue: coord})
            yerr_vals = ds[yerr].sel({hue: coord}) if yerr is not None else None

            handle = _plot_1d(
                _x=x_vals,
                _y=y_vals,
                _yerr=yerr_vals,
                _smooth_kwargs=copy.deepcopy(smooth_kwargs.get(pname, smooth_kwargs)),
                _ax=hlpr.ax,
                _label=f"{coord}",
                linestyle=linestyle if isinstance(linestyle, str) else linestyle[i],
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
        if x in ds.coords:
            x_vals = ds.coords[x]
        else:
            x_vals = ds[x]
        y_vals = ds[y]
        yerr_vals = ds[yerr] if yerr is not None else None

        _plot_1d(
            _x=x_vals,
            _y=y_vals,
            _yerr=yerr_vals,
            _ax=hlpr.ax,
            _smooth_kwargs=copy.deepcopy(smooth_kwargs.get(pname, smooth_kwargs)),
            _label=label,
            linestyle=linestyle,
            **plot_kwargs,
        )
