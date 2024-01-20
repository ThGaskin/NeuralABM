import numpy as np
import matplotlib.colors as mcolors
import xarray as xr
from dantro.plot.funcs.generic import make_facet_grid_plot
from dantro.plot.utils.color_mngr import ColorManager
from utopya.eval import PlotHelper, is_plot_func

from typing import Union

@make_facet_grid_plot(
    map_as="dataarray",
    encodings=("x", "y", "z", "col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    register_as_kind="surface",
    add_guide=False,
)
def surface(ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    _is_facetgrid: bool,
    x: str,
    y: str,
    cmap: Union[str, dict, mcolors.Colormap] = None,
    norm: Union[str, dict, mcolors.Normalize] = None,
    vmin: float = None,
    vmax: float = None,
    add_colorbar: bool = True,
    cbar_kwargs: dict = None,
    **plot_kwargs,
):
    if not hasattr(hlpr.ax, "zaxis"):
        raise AttributeError(
            "Missing z-axis! Did you set the "
            "projection (via `subplot_kws` or `setup_figure` helper)?"
        )

    cm = ColorManager(
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    shared_kwargs = dict(
        cmap=cm.cmap if cmap is not None else None,
        norm=cm.norm if norm is not None else None,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
    )

    # Get the coordinate data
    dim0 = ds.coords[x]
    dim1 = ds.coords[y]

    x_data, y_data = np.meshgrid(dim0, dim1)
    im = hlpr.ax.plot_surface(x_data, y_data, ds.data, **shared_kwargs, **plot_kwargs)

    if not _is_facetgrid and add_colorbar:
        # TODO This should read information from the FacetGrid's cbar_kwargs,
        #      which are also parsed there...
        cm.create_cbar(
            im,
            fig=hlpr.fig,
            ax=hlpr.ax,
            **(cbar_kwargs if cbar_kwargs else {}),
        )

    # FIXME Should do this via helper, but not working (see #82)
    # hlpr.provide_defaults("set_labels", x=x, y=y, z=z)
    hlpr.ax.set_xlabel(x)
    hlpr.ax.set_ylabel(y)
    hlpr.ax.set_zlabel(z)
