import xarray as xr
from dantro.plot.funcs.generic import make_facet_grid_plot

from utopya.eval import PlotHelper


@make_facet_grid_plot(
    map_as="dataset",
    encodings=("col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
    register_as_kind=True,
    overwrite_existing=True,
)
def bar(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    x: str,
    y: str,
    hue: str = None,
    _is_facetgrid: bool,
    **plot_kwargs,
):
    if "width" not in plot_kwargs:
        plot_kwargs["width"] = ds[x][1] - ds[x][0]

    hlpr.ax.bar(ds[x], ds[y], **plot_kwargs)


@make_facet_grid_plot(
    map_as="dataarray",
    encodings=("col", "row"),
    supported_hue_styles=("discrete",),
    hue_style="discrete",
    add_guide=False,
    register_as_kind=True,
    overwrite_existing=True,
    drop_kwargs=("x", "y"),
)
def hist(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    _is_facetgrid: bool,
    **plot_kwargs,
):
    hlpr.ax.hist(ds, **plot_kwargs)
