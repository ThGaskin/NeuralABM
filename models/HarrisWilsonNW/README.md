# The Harris-Wilson model of economic activity: network learning model

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

In this model, we infer the cost matrix  ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cmathbf%7BC%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN%20%5Ctimes%20M%7D).
The parameters ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha,%20%5Cbeta,%20%5Ckappa,%20%5Csigma)
are inferred in the sister `HarrisWilson` model.

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
You must specify the true values of ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Calpha,%20%5Cbeta,%20%5Ckappa,%20%5Csigma)
in the `Training` section of the config. This is to ensure the numerical solver runs with the same parameter
values as for the training data, especially if the training data was loaded externally. If you are generating
synthetic data before training, you can use YAML anchors so save yourself having to type the parameters out
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
You can of course use different values for ![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Csigma)
during training.

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
