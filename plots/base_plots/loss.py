import matplotlib.pyplot as plt
import numpy as np
import Utils


def plot_loss(loss_dict, *, labels: tuple=None, figsize: tuple = (Utils.textwidth, 0.35*Utils.textwidth)):

    """ Plots the training loss over time"""

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axs[0].plot(loss_dict['iteration'], loss_dict['training_loss'], color='black',
                label='Training loss' if labels is None else labels[0])
    axs[1].plot(loss_dict['iteration'], loss_dict['parameter_loss'], color='black',
                label='Parameter loss' if labels is None else labels[0])

    axs[1].set_xlabel('iteration')

    for ax in axs:
        ax.set_yscale('log')
        ax.legend(loc='upper right', fancybox=True, framealpha=0.)
        ax.grid(linewidth=0.5)

    plt.savefig(Utils.output_dir + '/loss')
    plt.close()
