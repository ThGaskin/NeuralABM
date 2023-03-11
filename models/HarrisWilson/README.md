# The Harris-Wilson model of economic activity

> **_Note_**: See the section on [Configuration sets](#configuration-sets) to see how to reproduce
> the plots from the [PNAS publication](https://www.pnas.org/doi/10.1073/pnas.2216415120).


### Model description
In the Harris-Wilson model, ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DN)
origin zones are connected to ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DM) destination
zones through a weighted, directed, complete bipartite network , i.e. each origin zone is connected to every destination zone.
Economic demand flows from the origin zones to the destination zones, which supply the demand.
Such a model is applicable for instance to an urban setting, the origin zones representing e.g.
residential areas, and the destination zones representing retail areas, shopping centres,
or other areas of consumer activity.

Let ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cmathbf%7BC%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%20%5Ctimes%20M%7D)
be the non-zero section of the full network adjacency matrix.
The network weights ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dc_%7Bij%7D)
quantify the convenience of travelling from origin zone ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Di)
to destination zone ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dj):
a low weight thus models a highly inconvenient route (e.g. due to a lack of public transport).
Each origin zone has a fixed demand ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DO_i).
The resulting cumulative demand at some destination zone ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Di)
is given by

> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DD_j%20=%20%5Csum_%7Bi=1%7D%5E%7BN%7D%20T_%7Bij%7D,)

![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DT_%7Bij%7D)
representing the flow of demand from ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Di)
to ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dj).
The model assumption is that this flow depends both on the size ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DW_j)
of the destination zone and the convenience of getting from ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Di)
to ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dj):

> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DT_%7Bij%7D%20=%20%5Cdfrac%7BW_j%5E%5Calpha%20c_%7Bij%7D%5E%5Cbeta%7D%7B%5Csum_%7Bk=1%7D%5EM%20W_k%5E%5Calpha%20c_%7Bik%7D%5E%5Cbeta%7D%20O_i.)

The parameters ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha)
and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cbeta) represent the
relative importance of size and convenience to the flow of demand
from ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Di)
to ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dj): high ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha)
means consumers value large destination zones (e.g. prefer larger shopping centres to smaller ones),
high ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cbeta)
means consumers place a strong emphasis on convenient travel to destination zones.
Finally, the sizes ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DW_j)
are governed by a system of ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DM)
coupled logistic equations:

> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cmathrm%7Bd%7DW_j%20=%20%5Cepsilon%20W_j(D_j%20-%20%5Ckappa%20W_j)%5Cmathrm%7Bd%7Dt%20&plus;%20%5Csigma%20W_j%20%5Ccirc%20%5Cmathrm%7Bd%7DB_j,)   [1]

with given initial conditions ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DW_j(t=0)%20=%20W_%7Bj,%200%7D).
Here, ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cepsilon)
is a responsiveness parameter, representing the rate at which destination zones can adapt to
fluctuations in demand, and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ckappa) models
the cost of maintaining a larger site per unit floor space (e.g. rent, utilities, etc.).
We recognise the logistic nature of the equations: the change in size is proportional to the size itself,
as well as to ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DD_j%20-%20%5Ckappa%20W_j).
A low value of ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ckappa) favours larger
destination zones (e.g. larger malls), a high cost favours smaller zones (e.g. local stores).
In addition, the model eq. [1] includes multiplicative noise with variance
![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Csigma%20%5Cgeq%200),
with ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ccirc) signifying Stratonovich integration.

In this model, we infer any of the four parameters ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha,%20%5Cbeta,%20%5Ckappa,%20%5Csigma).
The network is inferred in the sister `HarrisWilsonNW` model.

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
      init_mean: 0.1
      init_std: 0.01

      # Variance of the temporal fluctuations of the origin zone sizes
      temporal_std: 0.0

    # Settings for the initial destination size distribution
    init_dest_sizes:
      mean: 0.1
      std: 0.01

    # Settings for the initialisation of the weight matrix
    init_weights:
      mean: 1.2
      std: 1.2

  # Number of steps of the time series to use for training
  training_data_size: 1

```
`num_steps` generates a time series of that length, of which the last `training_data_size` steps
are used to train the model. If the model is in static equilibrium, the last frame of the time series
is sufficient.

### Specifying the parameters to learn
You can learn any of the parameters
![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha,%20%5Cbeta,%20%5Ckappa,%20%5Csigma).
Specify which parameters to learn in the `Training` entry:
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

- `Inequality`: sweeps over different values of ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha)
and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cbeta) and plots a heatmap of the 
inequality parameter ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cnu) (fig. 5) in 
the publication
- `London_dataset`: loads the London datasets, sweeps over two network metrics, and plots the marginal densities on the
parameters (fig. 10 in the publication)
- `Loss_landscape`: trains the model on synthetic data from multiple initialisations, and plots the resulting
loss landscape (fig. 6)
- `Marginals`: Plots the marginals and their peak widths for different levels of the noise in the 
data (fig. 7 & 8)
- `Performance_analysis`: plots the time to run one epoch and the loss after 6000 epochs as a function
of the origin size ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DN)
- `Sample_run`: runs and fits a small toy example

You can run these sets simply by calling

```commandline
utopya run HarrisWilson --cs name_of_cs
```