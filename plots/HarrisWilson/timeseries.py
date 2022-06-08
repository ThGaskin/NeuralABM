import matplotlib.pyplot as plt
import numpy as np
from typing import Any
import Utils


def plot_timeseries(data, *, x: Any = None, err: Any = None, figsize: tuple = (Utils.textwidth, Utils.textwidth),
                    labels: dict = None, output_dir: str, out_name: str = 'time_series'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linewidth=0.5)

    x = np.arange(0, data.shape[-1]) if x is None else x

    for i in range(len(data)):
        plt.plot(x, data[i], color=Utils.colors[i % len(Utils.colors)])
        if err is not None:
            plt.fill_between(x, data[i] - err[i], data[i] + err[i], alpha=0.7, fc='#2f7194')

    xlabel = r'$t$' if labels is None else labels['x']
    ylabel = r'$W_j$' if labels is None else labels['y']
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, rotation=0, labelpad=15)

    ax.autoscale(enable=True, axis='x', tight=True)
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)

    plt.savefig(output_dir + '/'+out_name)
    plt.close()
