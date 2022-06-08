import matplotlib.pyplot as plt
import Utils

def plot_time_loss_comp(data, *, n_iterations: int, figsize: tuple=(Utils.textwidth, 0.3*Utils.textwidth)):

    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.35})

    # Plot running time
    axs[0].errorbar(data['size'], data['time'], yerr=data['time_std'], color='black', marker='o',
                    markersize=2, elinewidth=0.7, capsize=2)
    axs[0].set_xlabel(r'$N+M$')
    axs[0].set_ylabel(fr'time to {n_iterations} iterations [s]')

    for ax in axs:
        ax.grid(True)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    axs[1].errorbar(data['size'], data['loss_final'], yerr=data['loss_final_std'],
                    label='Training loss', marker='o', color=Utils.color_dict['darkblue'],
                    markersize=2, elinewidth=0.7, capsize=2)
    axs[1].errorbar(data['size'], data['loss_frobenius'], yerr=data['loss_frobenius_std'],
                    label='Parameter loss', marker='^', color='black', markersize=2, elinewidth=0.7, capsize=2)
    axs[1].set_xlabel(r'$N+M$')
    axs[1].set_ylabel(fr'loss at {n_iterations} iterations')
    axs[1].set_yscale('log')
    axs[1].legend(ncol=1)

    plt.savefig(Utils.output_dir + '/time_loss_comparison.png')
    plt.close()