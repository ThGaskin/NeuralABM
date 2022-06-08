import logging
from utopya.plotting import is_plot_func, PlotHelper
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.ndimage

@is_plot_func(use_dag=True, required_dag_tags=('data',))
def plot_prob_density(data,
                      hlpr: PlotHelper,
                      *,
                      info_box_labels: list = None,
                      smooth_kwargs: dict = {},
                      **plot_kwargs):

    dset = data.pop('data')
    smooth, sigma = smooth_kwargs.pop('enabled', False), smooth_kwargs.pop('sigma', None)

    # 1d: plot a single probability density
    if not isinstance(dset, dict):

        y_vals = dset['prob']

        # Smooth the probability distribution, if set
        if smooth:
            y_vals = scipy.ndimage.gaussian_filter1d(y_vals, sigma, **smooth_kwargs)

        # Plot the distribution
        hlpr.ax.plot(dset['param1'], y_vals, **plot_kwargs)

    # 2d: plot stacked densities
    else:
        # Extract the datasets from the dictionary
        dsets = dset['data']

        for i, key in enumerate(dsets):

            y_vals = dsets[key]['prob']

            # Smooth the probability distribution, if set
            if smooth:
                s = sigma[i] if isinstance(sigma, list) else sigma
                y_vals = scipy.ndimage.gaussian_filter1d(y_vals, sigma, **smooth_kwargs)

            # Plot the probability distribution
            hlpr.ax.plot(dsets[key]['param1'], y_vals, label = f"{dset['param']} = {key}", **plot_kwargs)

    # Plot the textbox
    if info_box_labels:
        legend = hlpr.ax.legend([mlines.Line2D([], [], lw=0)] * len(data),
                            [info_box_labels[idx] + ' = ' + str(data[key]) for idx, key in enumerate(data)],
                            loc='best',
                            handlelength=0,
                            handleheight=0,
                            handletextpad=0)
        plt.gca().add_artist(legend)