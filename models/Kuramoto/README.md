# Kuramoto model of phase synchronisation

### Model description
The Kuramoto model describes phase-locking of $N$ coupled oscillators, each with phases $\varphi_i$ and eigenfrequencies $\omega_i$.
The oscillators are coupled via an adjacency matrix $\textbf{A} = (a_{ij})$ via the differential equation $$\alpha \dfrac{d^2 \varphi_i(t)}{dt^2} + \beta \dfrac{d\varphi_i(t)}{dt} = \omega_i + \kappa \sum_j a_{ij} \sin(\varphi_j - \varphi_i) + \sigma dB_i$$ where $\alpha$ is the inertia coefficient, $\beta$ the friction coefficient, $\kappa$ the coupling coefficient, and $B_i$ random noise with strength $\sigma$. In this model, we infer the adjacency matrix $\textbf{A}$ from observations of the phases $\varphi(t)$. The network is inferred from observations of its reponse to an initial perturbation: the coupled oscillators synchronise over time, until the phases are all in sync. These *reponse* dynamics allow us to infer $\textbf{A}$. Typically, a single observation time series $\textbf{T} = (\varphi(0), ..., \varphi(T))$ is insufficient to infer the entire network, and so neural network is trained on several independent observations simultaneously. However, in the British power grid configuration set, only a single time series is used (see below). The number of training sets used can be controlled via a `training_set_size` argument in the run configuration.

The model outputs e.g. the predicted and true network adjacency matrix, as well as predicted degree and triangle distributions with uncertainty quantification:

True and predicted adjacency matrix:
<img src="https://github.com/ThGaskin/NeuralABM/files/13854647/comp.pdf" width=100%>

Predicted degree distribution as a function of the noise $\sigma$ on the data:
<img src="https://github.com/ThGaskin/NeuralABM/files/13854606/degree_distribution.pdf" width=100%>

### Model parameters
The following are the default model parameters:
```yaml
Data:
  synthetic_data:

    # The number of nodes in the network
    N: 16

    # Network configuration
    network:
      mean_degree: 5
      type: random
      graph_props:
        is_directed: false
        is_weighted: true

    # Initial distribution of the eigenfrequencies
    eigen_frequencies:
      distribution: uniform
      parameters:
        lower: 1
        upper: 3
      time_series_std: 0.0 # Noise on the eigen frequencies time series

    # Initial distribution of the phases
    init_phases:
      distribution: uniform
      parameters:
        lower: 0
        upper: 6.283  # 2 pi

    # Noise variance on the training data
    sigma: 0.0

    # Length of each time series
    num_steps: 5

    # Number of individual time series
    training_set_size: 40

  # Time differential
  dt: 0.01

  # Inertia coefficient
  alpha: 0

  # Friction coefficient; must be strictly positive
  beta: 1

  # Coupling coefficient; must be strictly positive
  kappa: 1

```
`N` controls the number of vertices in the network; `network.mean_degree` and
`network.type` set the network mean degree and the network topology (see below).
The `eigen_frequencies` and `init_phases` keys set the initial distributions of the
node eigenfrequencies and the initial phases. The `distribution` key can be either `uniform` or
`normal`, and the `parameters` dictionary specifies the relevant parameters for the distribution
(`lower` and `upper` for uniform, and `mean` and `std` for normal distributions.)
The eigenfrequencies are a time series that fluctuate with variance `eigen_frequencies.time_series_std`);
set to 0 to use static eigenfrequencies.

`sigma` controls the noise of the data; `training_set_size` sets the number of time series to
generate from different initial conditions, and `num_steps` sets the number of steps per training set.

`alpha`, `beta`, and `kappa` set the coefficients of the differentials: set `alpha: 0` to
use a first-order model (default). `dt` sets the time differential.

### Controlling the network topology

The topology of the network can be controlled from the `synthetic_data.network` entry.
Set the topology via the `type` key; available options are: `random` (random network),
`star` (star network with one central node), `WattsStrogatz` (small-world network),
`BarabasiAlbert` (undirected scale-free graph), `BollobasRiordan` (directed scale-free graph),
and `regular` (regular lattice). Any additional parameters to the graph generation algorithm
must be supplied via a corresponding entry in the `graph_props` dictionary:

```yaml
network:
  graph_props:
    WattsStrogatz:
      p_rewire: 0.5 # Rewiring probability for the Watts-Strogatz algorithm
    BollobasRiordan:
      alpha: 0.4
      beta: 0.2
      gamma: 0.4 # These three parameters must sum to 1
```

### Loading data
You can also load training and network data from a file instead of generating it
synthetically. This is advantageous for instance when analysing real data, but also when
performing multiple training runs on the same dataset. To load data, use the following entry:

```yaml
Data:
  load_from_dir:
    network: path/to/h5file
    eigen_frequencies: path/to/h5file
    training_data: path/to/h5file
    copy_data: False
```
`copy_data` will copy the data over to the new `h5` File. It is set to true by default, but we recommend
turning this off if the loaded data is large, to avoid flooding the disc with duplicate datasets.

### Training the model
The training settings are controlled as described in the main README. When running the numerical solver during training,
you can set the noise level to use via the ``Training.true_parameters.sigma`` entry. By default, this is
zero, meaning the solver runs the noiseless version of the equations.

### Configuration sets
The following configuration sets are included in the model (some of which are located in the `Performance_analysis` 
folder)

- `Accuracy`: compares the prediction accuracy to that of OLS regression and MCMC, both for first- and second-order
equations
- `Computational_performance`: trains the model for 10 epochs for different network sizes, and plots the total
compute time. Also runs an MCMC for the same networks and plots compute times.
- `Convexity`: plot the error on the predicted degree distribution as a function of the rank of the Gram matrix of
observations; also plots the accuracy of the neural scheme compared to OLS regression and MCMC.
- `N_100_example`: infers a network with 100 nodes for different noise leves on the training data, and plots the results,
including the inferred degree and triangle distributions
- `N_1000_example`: same as for `N_100_example`, but for 1000 nodes.
- `Noise_performance`: trains the model and plots the Hellinger error, standard deviation, and relative entropy errors
on the degree and triangle distributions as a function of the noise on the training data
- `Powergrid`: UK powergrid example, with data given in `data/Kuramoto/UK_power_grid`

You can run these sets simply by calling

```commandline
utopya run Kuramoto --cs name_of_cs
```
