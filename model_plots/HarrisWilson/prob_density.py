import logging
from typing import Union

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
    x: str,
    y: str,
    yerr: str = None,
    hue: str = None,
    label: str = None,
    lw: Union[float, str, xr.DataArray] = None,
    alpha: Union[float, str, xr.DataArray] = None,
    suppress_labels: bool = False,
    info_box_labels: dict = None,
    smooth_kwargs: dict = {},
    **plot_kwargs,
):
    def _parse_plot_kwargs(param, coordinate) -> Union[float, None]:

        if isinstance(param, float):
            return param
        elif isinstance(param, str):
            return dset[param].sel({hue: coordinate})
        elif isinstance(param, xr.DataArray):
            return param.sel({hue: coordinate}).item()
        else:
            return None

    """Plots the marginal probability densities for a collection of datasets consisting of *different*
    x-values (param1) and associated probability values ('prob'). If specified, smooths the densities using
    a Gaussian kernel."""

    # Get the 'data' !dag_tag
    dset = data.pop("data") if isinstance(data, dict) else data

    # Get the smoothing properties
    smooth, sigma = smooth_kwargs.pop("enabled", False), smooth_kwargs.pop(
        "sigma", None
    )

    # Plot stacked lines
    _handles, _labels = [], []
    if hue:
        for i, coord in enumerate(dset.coords[hue].values):

            plot_kwargs.update(lw=_parse_plot_kwargs(lw, coord))
            plot_kwargs.update(alpha=_parse_plot_kwargs(alpha, coord))

            if x in dset.coords:
                x_vals = dset.coords[x]
            else:
                x_vals = dset[x].sel({hue: coord})

            y_vals = dset[y].sel({hue: coord})

            # Select the errorbands, if set
            y_vals_upper = (
                y_vals + dset[yerr].sel({hue: coord}) if yerr is not None else None
            )
            y_vals_lower = (
                y_vals - dset[yerr].sel({hue: coord}) if yerr is not None else None
            )

            # Smooth and normalise the probability distribution, if set
            if smooth:
                y_vals = scipy.ndimage.gaussian_filter1d(y_vals, sigma, **smooth_kwargs)

                if yerr is not None:
                    y_vals_upper = scipy.ndimage.gaussian_filter1d(
                        y_vals_upper, sigma, **smooth_kwargs
                    )
                    y_vals_lower = scipy.ndimage.gaussian_filter1d(
                        y_vals_lower, sigma, **smooth_kwargs
                    )

            # Plot the distribution
            (ebar,) = hlpr.ax.plot(x_vals, y_vals, **plot_kwargs)

            # Add errorbands, if given
            if yerr is not None:
                plot_kwargs.update(dict(alpha=0.5))
                fb = hlpr.ax.fill_between(
                    dset[x].sel({hue: coord}),
                    y_vals_lower,
                    y_vals_upper,
                    linewidth=0,
                    **plot_kwargs,
                )

            if not suppress_labels:
                _labels.append(f"{hue} = {coord}")
                _handles.append(ebar if yerr is None else (ebar, fb))

    else:

        x_vals, y_vals = dset[x], dset[y]

        # Select the errorbands, if set
        y_vals_upper = y_vals + dset[yerr] if yerr is not None else None
        y_vals_lower = y_vals - dset[yerr] if yerr is not None else None

        # Smooth the probability distribution, if set
        if smooth:
            y_vals = scipy.ndimage.gaussian_filter1d(y_vals, sigma, **smooth_kwargs)
            if yerr is not None:
                y_vals_upper = scipy.ndimage.gaussian_filter1d(
                    y_vals_upper, sigma, **smooth_kwargs
                )
                y_vals_lower = scipy.ndimage.gaussian_filter1d(
                    y_vals_lower, sigma, **smooth_kwargs
                )

        # Plot the distribution
        (ebar,) = hlpr.ax.plot(dset[x], y_vals, **plot_kwargs)

        # Add errorbands, if given
        if yerr is not None:
            plot_kwargs.update(dict(alpha=0.5))
            fb = hlpr.ax.fill_between(
                dset[x], y_vals_lower, y_vals_upper, linewidth=0, **plot_kwargs
            )

        if label is not None:
            _handles.append(ebar if yerr is None else (ebar, fb))
            _labels.append(label)

    hlpr.track_handles_labels(_handles, _labels)

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
            **info_box_labels.get("info_box_kwargs", {}),
        )
        plt.gca().add_artist(legend)
