# The Harris-Wilson model of economic activity

### Model description
In the Harris-Wilson model, $N$ origin zones are connected to $M$ destination zones through a weighted, directed, complete bipartite network, i.e. each origin zone is connected to every destination zone. Economic demand flows from the origin zones to the destination zones, which supply the demand. Such a model is applicable for instance to an urban setting, the origin zones representing e.g.
residential areas, and the destination zones representing retail areas, shopping centres, or other areas of consumer activity.

Let $\mathbf{C} = (c_{ij}) \in \mathbb{R}^{N \times M}$ be the non-zero section of the full network adjacency matrix. The network weights $c_{ij}$ quantify the convenience of travelling from origin zone $i$ to destination zone $j$: a low weight thus models a highly inconvenient route (e.g. due to a lack of public transport). Each origin zone has a fixed demand $O_i$. The resulting cumulative demand at some destination zone $j$ is given by $$D_j = \sum_{i=1}^N T_{ij}$$ $T_{ij}$ representing the flow of demand from $i$ to $j$ (the transport map, in optimal transport terms). The model assumption is that this flow depends both on the size $W_j$
of the destination zone and the convenience of getting from $i$ to $j$: $$T_{ij} = \dfrac{W_j^\alpha c_{ij}^\beta}{\sum_k W_k^\alpha c_{ij}^\beta}O_i$$

The parameters $\alpha$ and $\beta$ relative importance of size and convenience to the flow of demand
from $i$ to $j$: high $\alpha$ means consumers value large destination zones (e.g. prefer larger shopping centres to smaller ones),
high $\beta$ means consumers place a strong emphasis on convenient travel to destination zones.
Finally, the sizes $W_j$ are governed by a system of $M$ coupled logistic equations: $$dW_j = \epsilon W_j (D_j - \kappa W_j) dt + \sigma W_j \circ dB_j \quad [1]$$

with given initial conditions $W_j(t=0) = W_{j, 0}$. Here, $\epsilon$ is a responsiveness parameter, representing the rate at which destination zones can adapt to fluctuations in demand, and $\kappa$ models the cost of maintaining a larger site per unit floor space (e.g. rent, utilities, etc.). We recognise the logistic nature of the equations: the change in size is proportional to the size itself, as well as to $W_j$. A low value of $\kappa$ favours larger destination zones (e.g. larger malls), a high cost favours smaller zones (e.g. local stores). In addition, the model eq. [1] includes multiplicative noise with strength
$\sigma \geq 0$, with $\circ$ signifying Stratonovich integration.

In this model, we infer any of the four parameters $(\alpha, \beta, \kappa, \sigma)$. The network $\mathbf{C}$ is inferred in the sister `HarrisWilsonNW` model. See here the marginal densities on the parameters as the noise on the training data increases:

<img src="https://github.com/ThGaskin/NeuralABM/files/13855044/marginals.pdf" width=100%>

We can clearly see the width of the marginals increasing. Notice also the multimodality of the distributions on $\alpha$ and $\beta$, which have a second peak at $(\alpha=1, \beta=0)$.

### Model parameters
The following are the default model parameters:

```yaml
# Settings for the dataset, which is loaded externally or can be synthetically generated using the ABM
Data:
  synthetic_data:

    # Number of time series steps
    num_steps: 3000

    # Number of origin sizes
    N_origin: 100

    # Number of destination zone sizes
    N_destination: 10

    # Model parameters: size, convenience, cost, responsiveness, noise parameters
    alpha: 1.2
    beta: 4
    kappa: 2
    epsilon: 10
    sigma: 0

    # Time differential
    dt: 0.01

    # Settings for the initial origin size distribution
    origin_sizes:
      distribution: normal
      parameters:
        mean: 0.1
        std: 0.01

    # Settings for the initial destination size distribution
    init_dest_sizes:
      distribution: normal
      parameters:
        mean: 0.1
        std: 0.01

    # Settings for the initialisation of the weight matrix
    init_weights:
      distribution: normal
      parameters:
        mean: 1.2
        std: 1.2

  # Number of steps of the time series to use for training
  training_data_size: 1

```
`num_steps` generates a time series of that length, of which the last `training_data_size` steps
are used to train the model. If the model is in static equilibrium, the last frame of the time series
is sufficient.

### Specifying the parameters to learn
You can learn any of the parameters $(\alpha, \beta, \kappa, \sigma)$. Specify which parameters to learn in the `Training` entry:

```yaml
Training:
  to_learn: [alpha, beta, kappa, sigma]
```

For those parameters you do not wish to learn you must supply a true value:

```yaml
Training:
  to_learn: [alpha, beta]
  true_parameters:
    kappa: 5
    sigma: 0.0
```

### Loading data
Instead of generating synthetic data, you can also load data from an `.h5` File or `.csv` files.
For instance, you can load the origin zone, destination zone sizes, and the network all from separate `.csv` files:

```yaml
Data:
  load_from_dir:
    network: path/to/file
    origin_zones: path/to/file
    destination_zones: path/to/file
```
You can also load all three from a single `.h5` file:

```yaml
Data:
  load_from_dir: path/to/h5file
```
This `.h5` file must contain datasets called `origin_sizes`, `training_data`, and `network`.
If you first generate synthetic data using this model, you can thereafter point the config to the
`.h5` file. There is also plenty of synthetic data provided in the `data/` folder.

### Configuration sets
The following configuration sets are included in the model:

- `Inequality`: sweeps over different values of $\alpha$ and $\beta$ and plots a heatmap of the inequality parameter $\nu$ (fig. 5 in the publication)
- `London_dataset`: loads the London datasets, sweeps over two network metrics, and plots the marginal densities on the
parameters (fig. 10 in the publication)
- `Synthetic_example`: trains the model on synthetic data from multiple initialisations, and plots the resulting
loss landscape (fig. 6), marginal plots, joint distributions, and initial value distribution
- `Marginals_over_noise`: Plots the marginals for different levels of the noise in the data

You can run these sets simply by calling

```commandline
utopya run HarrisWilson --cs name_of_cs
```
