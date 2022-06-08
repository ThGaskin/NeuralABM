import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import Utils

def plot_heatmap(data, *, extent, figsize: tuple = (Utils.textwidth, Utils.textwidth), labels: dict=None, norm = None,
                 vmin=None, vmax=None):

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap=Utils.colorbar,
                   norm=norm, vmin=vmin, vmax=vmax)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="5%")
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)

    if labels is not None:
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'], rotation=0, labelpad=10)
        cbar.set_label(labels['z'])

    for loc in ['top', 'bottom', 'right', 'left']:
        ax.spines[loc].set_visible(False)

    plt.savefig(Utils.output_dir + '/heatmap')
    plt.close()
