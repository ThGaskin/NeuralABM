import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.stats as stats
import numpy as np
from typing import Any
import Utils


def plot_param_predictions(data, true_params, *,
                           x,
                           averaging_window: float = 0.1,
                           prob_densities: Any = None,
                           figsize: tuple = (Utils.textwidth, 0.5 * Utils.textwidth)):

    param_labels = [r'$\hat{'+f'\{p}'+r'}$' for p in true_params.keys()]
    labels = [fr'$\{p}$' for p in true_params.keys()]

    if prob_densities is not None:

        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3,2], 'wspace': 0.1}, sharey=True)
        ax0, ax1 = axs[0], axs[1]
        ax1.grid(True, linewidth=0.5)

    else:
        fig, ax0 = plt.subplots(figsize=figsize)

    ax0.grid(True, linewidth=0.5)

    averaging_window = max(1, int(averaging_window * data.shape[-1]))
    x = x[averaging_window:]

    for idx, col in enumerate(data):
        means, stddevs = [], []
        for j in range(averaging_window, len(data[col])):
            means.append(np.mean(data[col][j - averaging_window:j]))
            stddevs.append(np.std(data[col][j - averaging_window:j]))
        means = np.array(means)
        stddevs = np.array(stddevs)
        ax0.fill_between(x, means - stddevs, means + stddevs, alpha=0.5)
        ax0.plot(x, means, color=Utils.colors[idx], label=param_labels[idx])

    ax0.set_xlabel(r'iteration')
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    A = list(plt.yticks()[0]) + list(true_params.values())
    B = [str(a) for a in A][:-len(true_params.values())] + labels
    plt.yticks(A, B)
    for _ in range(len(data.columns)):
        ax0.get_yticklabels()[-len(data.columns)+_].set_color(Utils.colors[_])

    ax0.set_xlim(x[0], )

    # Stochastic case: also add probability distributions
    if prob_densities is not None:

        x2 = np.linspace(ax0.get_ylim()[0], ax0.get_ylim()[1], 2000)
        ax1.set_ylim(ax0.get_ylim()[0], ax0.get_ylim()[1])

        for _ in range(len(prob_densities)):
            mu, std = np.mean(prob_densities[_]), np.std(prob_densities[_])
            ax1.plot(stats.norm.pdf(x2, mu, std), x2, color=Utils.colors[_], label=param_labels[_])
            ax1.fill_betweenx(x2, 0, stats.norm.pdf(x2, mu, std), alpha=0.5)

        ax0.autoscale(enable=True, axis='x', tight=True)
        ax1.set_xlim(0, )
        for loc in ['top', 'right']:
            ax1.spines[loc].set_visible(False)


    for loc in ['top', 'right']:
        ax0.spines[loc].set_visible(False)

    if prob_densities is not None:
        ax1.legend(loc='best', ncol=len(labels), fancybox=True, framealpha=0.1,)
    else:
        ax0.legend(loc='best', ncol=len(labels), fancybox=True, framealpha=0.1,)

    # plot the true parameter lines
    for idx, p in enumerate(true_params):
        ax0.plot(x, true_params[p] * np.ones_like(x), linewidth=1.0, linestyle='dotted',
                 color='black', zorder=2, alpha=0.5)
        if prob_densities is not None:
            ax1.plot(np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 100),
                     true_params[p] * np.ones(100), color='black', linewidth=1.0, linestyle='dotted', zorder=2,
                     alpha=0.5)

    ax0.set_xscale('log')
    plt.savefig(Utils.output_dir + '/predictions')
    plt.close()