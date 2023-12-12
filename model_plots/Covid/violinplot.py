import copy

import numpy as np
import scipy.integrate
import scipy.ndimage
import xarray as xr
from dantro.plot.funcs.generic import make_facet_grid_plot

from utopya.eval import PlotHelper, is_plot_func


@make_facet_grid_plot(
    map_as="dataset",
    encodings=("x", "y", "hue", "col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
)
def violin_plot(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    _is_facetgrid: bool,
    x: str,
    y: str,
    hue: str,
    add_legend: bool = True,
    show_means: bool = True,
    show_modes: bool = True,
    format_y_label: bool = False,
    mean_kwargs: dict = dict(s=15, color="#48675A", lw=0.3, edgecolor="#3D4244"),
    mode_kwargs: dict = dict(s=15, color="#F5DDA9", lw=0.3, edgecolor="#3D4244"),
    smooth_kwargs: dict = {},
    **plot_kwargs,
):
    """Plots a violinplot of different datasets. The ``hue`` dimension is plotted in an alternating
    fashion on the left and right sides of the plot, although this renders the plot somewhat pointless if the length
    of the ``hue`` dimension is greater than 2. Means and modes of the modes can also be shown as discrete points.

    :param ds: ``xr.Dataset`` of data values
    :param hlpr: ``PlotHelper`` instance
    :param x: variable to plot on the x dimension
    :param y: variable to plot on the y dimension
    :param hue: variable to alternately plot on the left and right side of the y-axis
    :param add_legend: passed to ``xr.facet_grid``
    :param show_means: (optional) whether to show the means of the distributions
    :param show_modes: (optional) whether to show the modes of the distribution
    :param format_y_label: (optional) whether to format the y-labels to match the Berlin SEIRD publication style
        ``$\\lambda_{\rm X}$``.
    :param mean_kwargs: plot_kwargs for the mean dots, passed to ``ax.scatter``
    :param mode_kwargs plot_kwargs for the mean dots, passed to ``ax.scatter``
    :param plot_kwargs: plot_kwargs for the distribution, passed to ``ax.fillbetweenx``
    """

    def _plot_1d(
        _x, _y, _yfactor, *, _smooth_kwargs: dict = {}, label: str, **_plot_kwargs
    ):
        """Plots a single parameter density and smooths the marginal. Returns the artists for the legend."""
        smooth, sigma = _smooth_kwargs.pop("enabled", False), _smooth_kwargs.pop(
            "smoothing", None
        )
        # Smooth the y values, if given
        if smooth:
            _y = scipy.ndimage.gaussian_filter1d(_y, sigma, **_smooth_kwargs)

        _handle = hlpr.ax.fill_betweenx(
            _x,
            _yfactor * _y,
            np.zeros(len(_y)),
            alpha=0.6,
            lw=2,
            label=label,
            **_plot_kwargs,
        )

        if show_means:
            mean_x = scipy.integrate.trapezoid(_x * _y, _x)
            mean_y = _y.data[np.argmin(np.abs(_x - mean_x).data)]
            _mean_handle = hlpr.ax.scatter(
                _yfactor * mean_y, mean_x, **mean_kwargs, label="Mean"
            )
        else:
            _mean_handle = None
        if show_modes:
            mode_x, mode_y = _x[_y.argmax()], np.max(_y)
            _mode_handle = hlpr.ax.scatter(
                _yfactor * mode_y, mode_x, **mode_kwargs, label="Mode"
            )
        else:
            _mode_handle = None
        return _handle, _mean_handle, _mode_handle

    if "parameter" in list(ds.coords):
        pname = ds.coords["parameter"].values.item()
    else:
        pname = list(ds.coords.keys())[0]

    _handles, _labels = [], []
    for i, coord in enumerate(ds.coords[hue].values):
        if x in ds.coords:
            x_vals = ds.coords[x]
        else:
            x_vals = ds[x].sel({hue: coord})
        y_vals = ds[y].sel({hue: coord})
        _handle, _mean_handle, _mode_handle = _plot_1d(
            x_vals,
            y_vals,
            ((-1) ** (i + 1)),
            _smooth_kwargs=copy.deepcopy(smooth_kwargs.get(pname, smooth_kwargs)),
            label=hue,
            **plot_kwargs,
        )
        _handles.append(_handle)
        _labels.append(f"{coord}")

    if _mean_handle:
        _handles.append(_mean_handle)
        _labels.append("Mean")
    if _mode_handle:
        _handles.append(_mode_handle)
        _labels.append("Mode")

    if not _is_facetgrid:
        if add_legend:
            hlpr.ax.legend(_handles, _labels, title="")
    else:
        if add_legend:
            hlpr.track_handles_labels(_handles, _labels)
            hlpr.provide_defaults("set_figlegend", title="")

    if format_y_label:
        y_label = (
            r"$\lambda_{\rm "
            + ds.coords["parameter"].item()[2:].replace("_", ",")
            + "}$"
        )
        hlpr.provide_defaults("set_labels", y={"label": y_label})

    # Positive values on both axes
    hlpr.ax.set_xticks(
        hlpr.ax.get_xticks()[1:], labels=np.round(np.abs(hlpr.ax.get_xticks())[1:], 2)
    )
