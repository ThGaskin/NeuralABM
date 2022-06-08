import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import Utils


def plot_trajectories_3d(data, *, view: dict=None,  xlim: tuple=None, ylim: tuple=None, zlim: tuple=None,
                         true_params: dict=None, figsize: tuple = (0.5*Utils.textwidth, 0.5*Utils.textwidth)):

    def color_sorter(val):
        if true_params is not None:
            if abs(val-0.) < 0.05:
                return Utils.color_dict['red']
            elif (abs(val-true_params['beta']) < 0.15):
                return Utils.color_dict['darkblue']
            else:
                return 'black'
        else:
            return  Utils.color_dict['darkblue']

    fig, ax = plt.figure(figsize=figsize), plt.axes(projection='3d')
    if view is not None:
        ax.view_init(elev=view['elev'], azim=view['azim'])

    for i in range(len(data)):

        x, y, z = data[data.columns[0]][i], data[data.columns[1]][i], data[data.columns[2]][i]

        # Plot the final points
        #ax.scatter(x[-1], y[-1], z[-1], color=Utils.color_dict['green'], s=16, alpha=1.0, linewidth=0)

        # Plot the initial points
        ax.scatter(x[0], y[0], z[0], c=color_sorter(y[-1]), s=21, alpha=1.0, linewidth=0, cmap=Utils.colorbar)

        # Plot the trajectories
        ax.plot3D(x, y, z, color=Utils.color_dict['grey'], alpha=0.6, linewidth=0.3)
    
    # Plot the true values 
    if true_params is not None:
        ax.scatter(true_params['alpha'], true_params['beta'], true_params['kappa'], color='#ec7070', s=7)
        
    # Plot labels
    ax.set_xlabel(fr'$\{data.columns[0]}$')
    ax.set_ylabel(fr'$\{data.columns[1]}$')
    ax.set_zlabel(fr'$\{data.columns[2]}$', rotation=0)
    ax.zaxis.set_rotate_label(False)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_xlim(zlim)
    
    # Add a legend
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              label=r'$\hat{\beta}_F \approx $'+fr"$ {true_params['beta']}$",
                              markerfacecolor=Utils.color_dict['darkblue'], markersize=6),
                       Line2D([0], [0], marker='o', color='w',
                              label=r'$\hat{\beta}_F \approx 0$', markerfacecolor=Utils.color_dict['red'], markersize=6)]
    ax.legend(handles=legend_elements, loc='upper center', ncol=2)

    plt.savefig(Utils.output_dir+'/trajectories_3d.png')
    plt.close()