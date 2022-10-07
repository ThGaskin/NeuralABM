import sys
import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.ndimage
import xarray as xr

from utopya.plotting import is_plot_func, PlotHelper

log = logging.getLogger(__name__)


@is_plot_func(use_dag=True)
def plot_destination_size_predictions(data: xr.Dataset,
                      hlpr: PlotHelper,
                      *,
                      hue: str = None,
                      info_box_labels: dict = None,
                      **plot_kwargs):

    """ Plots the predictions of destination sizes against the destination size data  """
    # Get the data
    x_predictions = data.pop('x_predictions')
    x_data = data.pop('x_data')
    
    # print(x_data.coords[hue].values)
    # Plot predictions
    if hue:
        for coord in x_data.coords[hue].values:
            y_vals = x_data.sel({hue: coord})
            x_vals = x_predictions.sel({hue: coord})
            # Plot the scatter plot
            hlpr.ax.scatter(x_vals.to_array(), y_vals.to_array(), label=coord, **plot_kwargs)
    else:
        # Plot the distribution
        hlpr.ax.scatter(x_predictionsto_array(), x_data.to_array(), label=coord, **plot_kwargs)

    # Plot the textbox, if given, using the remaining !dag_tags, which should be floats
    if info_box_labels:
        legend = hlpr.ax.legend([mlines.Line2D([], [], lw=0)] * len(data),
                            [info_box_labels[key] + ' = ' + str(data[key]) for key in list(data.keys())],
                            loc='best',
                            handlelength=0,
                            handleheight=0,
                            handletextpad=0)
        plt.gca().add_artist(legend)