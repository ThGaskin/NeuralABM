# The Harris-Wilson model of economic activity: network learning model

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

In this model, we infer the cost matrix $\mathbf{C}$. The parameters $(\alpha, \beta, \kappa, \sigma)$ are inferred in the sister `HarrisWilson` model.

### Model parameters
The following are the default model parameters:

```yaml
# Settings for the dataset, which is loaded externally or can be synthetically generated using the ABM
Data:
  synthetic_data:

    # Number of steps in the synthetic time series
    num_steps: 500

    # Origin zone and destination zone size
    N_origin: 10
    N_destination: 5

    # Size, convenience, cost, noise, and responsiveness parameters
    alpha: 1.2
    beta: 4
    kappa: 2
    sigma: 0
    epsilon: 10

    # Time differential
    dt: 0.001

    # Initialisation of the network weights
    init_network_weights:
      distribution: normal
      parameters:
        mean: 1.2
        std: 1.2

    # Initialisation of the origin sizes
    init_origin_sizes:
      distribution: normal
      parameters:
        mean: 1
        std: 0.2

    # Fluctuations of the origin zones over time
    origin_size_std: 0.05

    # Initialisation of the destination sizes
    init_dest_sizes:
      distribution: normal
      parameters:
        mean: 1
        std: 0.2

  # Number of independently training sets to use
  training_set_size: 1

  # Number of training steps to use
  num_training_steps: 300
```

`num_steps` generates a time series of that length, of which the last `num_training_steps` steps
are used to train the model. `training_set_size` sets the number of time series to
generate from different initial conditions, and `num_steps` sets the number of steps per training set.

### Specifying the true parameters
You must specify the true values of $(\alpha, \beta, \kappa, \sigma)$ in the `Training` section of the config. This is to ensure the numerical solver runs with the same parameter values as for the training data, especially if the training data was loaded externally. If you are generating synthetic data before training, you can use YAML anchors so save yourself having to type the parameters out
again:

```yaml
Data:
  synthetic_data:
    alpha: &alpha 1.2
    beta: &beta 3
    kappa: &kappa 8
    epsilon: &epsilon 3
    sigma: &sigma 0
    dt: &dt 0.001
Training:
  true_parameters:
    alpha: *alpha
    beta: *beta
    kappa: *kappa
    epsilon: *epsilon
    sigma: *sigma
    dt: *dt
```
This will automatically use the parameters you used to generate the training data during the neural net training run.
You can of course use different values for $\sigma$ during training.

### Loading data
Instead of generating synthetic data, you can also load data from an `.h5` File. Specify the paths to the
origin sizes, destination sizes, and network files from the `load_from_dir` entry of the config:

```yaml
Data:
  load_from_dir:
    network: path/to/file.h5
    origin_zones: path/to/file.h5
    destination_zones: path/to/file.h5
```
You do not need to provide the paths to all three -- any missing files will be synthetically generated using
the settings from the `synthetic_data` entry.

### Configuration sets
The following configuration sets are included in the model:

- `London_dataset`: trains on a synthetic dataset using the London dataset as an initial condition.
The data is provided in the `data/HarrisWilsonNW/` folder. You can generate it by calling
```commandline
python3 models/HarrisWilsonNW/cfgs/London_dataset/generate_London_data.py
```
This will automatically generate data using the settings in the `London_dataset` configuration set and write it
to the `data/HarrisWilonNW` folder (it will overwrite any previous files!)
You can run this set simply by calling

```commandline
utopya run HarrisWilsonNW --cs London_dataset
```
