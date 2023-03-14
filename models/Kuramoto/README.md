# Kuramoto model of phase synchronisation

---
### Model description
The Kuramoto model describes phase-locking of ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DN) coupled oscillators,
each with phases ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cvarphi_i) and eigenfrequencies
![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Comega_i).
The oscillators are coupled via
an adjacency matrix ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cmathbf%7BA%7D%20=%20(a_%7Bij%7D))
via the differential equation

> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cdfrac%7B%5Cmathrm%7Bd%7D%5Cvarphi_i%7D%7B%5Cmathrm%7Bd%7Dt%7D%20=%20%5Comega_i%20&plus;%20%5Ckappa%5Csum_j%20a_%7Bij%7D%20%5Csin(%5Cvarphi_j%20-%5Cvarphi_i)%20&plus;%20%5Csigma%20%5Cmathrm%7Bd%7DB_i)

where ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ckappa) is a network coupling
constant and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DB_i) is random noise with variance
![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Csigma). The model also exists in a second-order
version,

> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cgamma%20%5Cdfrac%7B%5Cmathrm%7Bd%7D%5E2%5Cvarphi_i%7D%7B%5Cmathrm%7Bd%7Dt%5E2%7D%20&plus;%20%5Cdfrac%7B%5Cmathrm%7Bd%7D%5Cvarphi_i%7D%7B%5Cmathrm%7Bd%7Dt%7D%20=%20%5Comega_i%20&plus;%20%5Csum_j%20a_%7Bij%7D%20%5Csin(%5Cvarphi_j-%5Cvarphi_i)%20&plus;%20%5Cmathrm%7Bd%7DB_i)

In this model, we infer the network from observations of the phases.

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
      time_series: false
      time_series_std: 0.0

    # Initial distribution of the phases
    init_phases:
      distribution: uniform
      parameters:
        lower: 0
        upper: 6.283

    # Noise variance on the training data
    sigma: 0.0

    # Length of time series
    num_steps: 5

    # Number of individual time series
    training_set_size: 40

  # Time differential
  dt: 0.01

  # Dampening coefficient (second order only)
  gamma: 1

  # Network coupling scaling value
  kappa: 1

# Whether to use a second order model
second_order: False
```
`N` controls the number of vertices in the network; `network.mean_degree` and
`network.type` set the network mean degree and the network topology (see below).
The `eigen_frequencies` and `init_phases` keys set the initial distributions of the
node eigenfrequencies and the initial phases. The `distribution` key can be either `uniform` or
`normal`, and the `parameters` dictionary specifies the relevant parameters for the distribution
(`lower` and `upper` for uniform, and `mean` and `std` for normal distributions.)
The eigenfrequencies can be static or a time series (set `eigen_frequencies.time_series` to `true`
and specify the variance of the fluctuations via `eigen_frequencies.time_series_std`).

`sigma` controls the noise of the data; `training_set_size` sets the number of time series to
generate from different initial conditions, and `num_steps` sets the number of steps per training set.

`second_order` determines whether to use a second-order Kuramoto model; if set, use the `gamma` key to set
the coefficient on the second derivative. `dt` sets the time differential, and `kappa` controls
the network coupling coefficient.

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
