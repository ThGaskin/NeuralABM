import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import Utils

def plot_loss_density(data, *, parameters: list, true_params: dict=None, view: dict=None,
                      xlim: tuple=None, ylim: tuple=None, zlim: tuple=None,
                      figsize: tuple = (0.5*Utils.textwidth, 0.5*Utils.textwidth)):

    dim = len(parameters)
    if dim == 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, linewidth=0.5)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    else:
        fig, ax = plt.figure(figsize=figsize), plt.axes(projection='3d')
        if view is not None:
            ax.view_init(elev=view['elev'], azim=view['azim'])

    if dim == 1:
        zipped_data = np.transpose(list(zip([item for sublist in data[parameters[0]] for item in sublist],
                        [item for sublist in data['loss_final'] for item in sublist])), (1, 0))
        ax.scatter(zipped_data[0], -np.log10(zipped_data[1]), c=zipped_data[1], alpha=0.9, linewidth=0,
                   cmap=Utils.colorbar, norm=LogNorm(), s=5)

    elif dim == 2:
        zipped_data = np.transpose(list(zip([item for sublist in data[parameters[0]] for item in sublist],
                                            [item for sublist in data[parameters[1]] for item in sublist],
                                            [item for sublist in data['loss_final'] for item in sublist])), (1, 0))
        ax.scatter(zipped_data[0], zipped_data[1], -np.log10(zipped_data[2]), c=zipped_data[2], alpha=0.9, linewidth=0,
                   cmap=Utils.colorbar, norm=LogNorm(), s=5)
        num, t_max = 46, 5900
        ax.plot(data['alpha'][num][:-t_max], data['beta'][num][:-t_max], -np.log10(data['loss_final'][num][:-t_max]), c='black',
                alpha=0.9, linewidth=0.5, linestyle='dotted', zorder=10)

    elif dim == 3:
        zipped_data = np.transpose(list(zip([item for sublist in data[parameters[0]] for item in sublist],
                                            [item for sublist in data[parameters[1]] for item in sublist],
                                            [item for sublist in data[parameters[2]] for item in sublist],
                                            [item for sublist in data['loss_final'] for item in sublist])), (1, 0))
        ax.scatter(zipped_data[0], zipped_data[1], zipped_data[2], c=zipped_data[3], alpha=0.9, linewidth=0,
                   cmap=Utils.colorbar, norm=LogNorm(), s=5)

    if true_params:
        if dim == 1:
            plt.axvline(x=true_params[parameters[0]], color=Utils.color_dict['red'], linewidth=0.8, linestyle='dotted')
        elif dim == 2:
            max_val = -np.log10(np.amin(zipped_data[2]))
            ax.plot([true_params[parameters[0]], true_params[parameters[0]]],
                    [true_params[parameters[1]], true_params[parameters[1]]],
                    [-0.17, 1.1*max_val], color=Utils.color_dict['red'], linewidth=1)
            ax.plot([1, 1],
                    [0, 0],
                    [-0.17, 1.1*max_val], color=Utils.color_dict['green'], linewidth=1)
        elif dim == 3:
            ax.scatter(true_params[parameters[0]], true_params[parameters[1]], true_params[parameters[2]],
                       color=Utils.color_dict['red'], s=10)

    # Set the labels
    if dim == 1:
        ax.set_xlabel(fr'$\{parameters[0]}$')
        ax.set_ylabel(r'$-\log_{10}(J)$', rotation=90)
    elif dim == 2:
        ax.set_xlabel(fr'$\{parameters[0]}$')
        ax.set_ylabel(fr'$\{parameters[1]}$')
        ax.set_zlabel(r'$-\log_{10}(J)$', rotation=90)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlim(0, )
    elif dim == 3:
        ax.set_xlabel(fr'$\{parameters[0]}$')
        ax.set_ylabel(fr'$\{parameters[1]}$')
        ax.set_zlabel(fr'$\{parameters[2]}$')
        ax.zaxis.set_rotate_label(False)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_xlim(zlim)

    plt.savefig(Utils.output_dir + f"/density_{dim}d_{'_'.join(parameters)}.png", transparent=True)
    plt.close()

def plot_prob_density(dataset, *, parameter: str, true_params: dict=None, view: dict=None,
                      xlim: tuple=None, ylim: tuple=None, zlim: tuple=None,
                      labels: list=None,
                      figsize: tuple = (0.5*Utils.textwidth, 0.5*Utils.textwidth)):

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linewidth=0.5)
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)

    for idx, data in enumerate(dataset):
        all_data = list(zip([item for sublist in data[parameter] for item in sublist],
                            [-np.log10(item) for sublist in data['loss_final'] for item in sublist]))

        point_pairs = sorted(all_data, key=lambda x: x[0])
        bins = np.linspace(point_pairs[0][0], point_pairs[-1][0], 100)
        vals_split = [[point_pairs[0]]]
        bin_no = 1
        for point in point_pairs:
            if bin_no != len(bins):
                if point[0] <= bins[bin_no]:
                    if point[1] > vals_split[-1][0][1]:
                        vals_split[-1][0] = point
                else:
                    bin_no += 1
                    vals_split.append([point])
            else:
                if point[0] <= bins[-1]:
                    if point[1] > vals_split[-1][0][1]:
                        vals_split[-1][0] = point
        vals_split = np.transpose(np.reshape(vals_split, (len(vals_split), 2)), (1, 0))
        vals_split[1] = vals_split[1]-min(vals_split[1])


        # dx = float(vals_split[0][-1] - vals_split[0][0]) / len(vals_split[0])
        # gx = np.arange(0, vals_split[0][-1], dx)
        # gaussian = np.exp(-(gx / 0.05) ** 2 / 2)
        # result = np.convolve(np.exp(vals_split[1]), gaussian, mode="full")
        #ax.plot(np.linspace(0, 2*vals_split[0][-1], len(result)), result/np.sum(result), linewidth=0.8)
        ax.plot(vals_split[0], np.exp(vals_split[1]) / np.sum(np.exp(vals_split[1])), linewidth=1,
                label=labels[idx] if labels is not None else None)
    if parameter == 'beta':
        ax.set_xlim(3, 5)
    else:
        ax.set_xlim(ax.get_xlim()[0], vals_split[0][-1]+1)
    ax.set_xlabel(fr'$\{parameter}$')
    #ax.set_ylabel(r'$-\log_{10}(J) \star K_\sigma $', rotation=90)
    ax.set_ylabel(r'$e^{-\log_{10}(J)}/Z$', rotation=90)
    ax.set_yscale('log')
    if labels is not None:
        ax.legend(loc='upper center', ncol=len(labels), fancybox=True, framealpha=0.,
                  bbox_to_anchor=(0.5, 1.2), fontsize=9)

    if true_params:
        plt.axvline(x=true_params[parameter], color=Utils.color_dict['red'], linewidth=0.8,
                    linestyle='dotted')

    plt.savefig(Utils.output_dir + f"/prob_density_1d_{parameter}.png")
    plt.close()