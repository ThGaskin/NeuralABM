import matplotlib.colors as mcolors
from typing import Union
from utopya.plotting import is_plot_func, PlotHelper
from dantro.plot.utils import ColorManager


@is_plot_func(use_dag=True, supports_animation=True)
def scatter(*,
            hlpr: PlotHelper,
            data: dict,
            cmap: Union[str, dict, mcolors.Colormap] = None,
            norm: Union[str, dict, mcolors.Normalize] = None,
            vmin: float = None,
            vmax: float = None,
            add_colorbar: bool = True,
            cbar_kwargs: dict = None,
            frames_isel: list = None,
            **plot_kwargs):
    """ Plots an animation of the agents in space, colored by their 'kind'.

    :param hlpr: the PlotHelper
    :param data: the data to plot
    :param cmap (Union[str, dict, matplotlib.colors.Colormap], optional): The
           colormap, passed to the :py:class:`~dantro.plot.utils.color_mngr.ColorManager`.
    :param norm: the cmap norm
    :param vmin (float, optional): The lower bound of the color-mapping.
           Ignored if norm is *BoundaryNorm*.
    :param vmax (float, optional): The upper bound of the color-mapping.
           Ignored if norm is *BoundaryNorm*.
    :param add_colorbar (bool, optional): Whether to add a colorbar
    :param cbar_kwargs (dict, optional): Arguments for colorbar creation.
    :param **plot_kwargs: Passed on to :py:func:`matplotlib.axes.Axes.scatter`

    """
    # Set up the dantro ColorManager
    cm = ColorManager(
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    # Plot the image
    im = hlpr.ax.scatter(data['data'][{'time': 0, 'coords': 0}]['position'],
                         data['data'][{'time': 0, 'coords': 1}]['position'],
                         c=data['data'][{'time': 0}]['kinds'],
                         cmap=cm.cmap,
                         norm=cm.norm,
                         **plot_kwargs)

    hlpr.ax.set_aspect("equal")

    # Add the Colorbar, if specified
    if add_colorbar:
        cm.create_cbar(
            im,
            fig=hlpr.fig,
            ax=hlpr.ax,
            **(cbar_kwargs if cbar_kwargs else {}),
        )

    frames_isel = frames_isel if frames_isel is not None else data['data'].coords['time']

    # Animation update function
    def update():
        for t in frames_isel:
            im.set_offsets(data['data']['position'].sel({'time': t}))
            im.set_array(data['data']['kinds'].sel({'time': t}).values.flatten())
            yield

    # Register the animation
    hlpr.register_animation_update(update)
