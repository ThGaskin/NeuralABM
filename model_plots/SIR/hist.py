import xarray as xr
from dantro.plot.funcs.generic import make_facet_grid_plot

from utopya.eval import PlotHelper, is_plot_func


@make_facet_grid_plot(
    map_as="dataset",
    encodings=("col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
)
def plot_bar(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    x: str,
    y: str,
    hue: str = None,
    _is_facetgrid: bool,
    **plot_kwargs,
):

    hlpr.ax.bar(ds[x], ds[y], **plot_kwargs)
