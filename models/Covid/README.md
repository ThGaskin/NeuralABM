# MODUS-Covid ODE model

### Model description

This is the ODE model described in our paper on [neural parameter calibration for Covid](https://arxiv.org/abs/2312.03147)
as well as the [paper](https://pubmed.ncbi.nlm.nih.gov/33887760/) by Wulkow et al. It is a compartmental ODE model
of epidemics containing the following compartments:
- Susceptible, with associated transition parameter $k_S$
- Exposed, with associated transition parameter $k_E$
- Infected, with associated transition parameter $k_I$
- Recovered, with associated transition parameter $k_R$
- Symptomatic, with associated transition parameter $k_{SY}$
- Hospitalised, with associated transition parameter $k_H$
- Deceased, with associated transition parameter $k_D$
- Quarantined (subdivided into S, E, I compartments), with associated transition parameter $k_Q$
- Contacted by the contact tracing agency, with associated transition parameter $k_{CT}$

Note that these parameters were donated with a $\lambda$ symbol in the publication.
This model learns the transition parameters between these compartments from data. Transition parameters
can be time-dependent, and the model allows specifying arbitrary intervals for any subset of the
model parameters. The transition parameter $k_Q$ is not inferred, but rather calculated from $k_CT$ and
$CT$ directly via $$\dfrac{\mathrm{d}k_Q}{\mathrm{d}t}=k_q k_{CT} CT$$

It thus cannot be learned. The model can be run by calling

```commandline
utopya run Covid
```

The model calibrates the time series for each of the compartments and produces a prediction by drawing parameter samples from the estimated joint distribution of the parameters:

<img src="https://github.com/ThGaskin/NeuralABM/files/13854843/densities_from_joint.pdf" width=100%>

The data can be divided into a training (calibration) and test (prediction) section; the neural network is then trained on the training section and a prediction generated on the test section:

<img src="https://github.com/ThGaskin/NeuralABM/files/13854838/all.pdf" width=100%>

Here, the red period is the training section, and the blue shaded area the test period. This can be controlled via the `Data.training_set_size` key (see below).

### Model parameters
The following are the default parameters for the MODUS-Covid model:
```yaml
Data:
  synthetic_data:

    # ODE model parameters for generating synthetic data
    k_S: 0.4
    k_E:  3.4
    k_I:  1.0
    k_R: 0.8
    k_SY:  0.5
    k_H:  0.01
    k_C:  0.5
    k_D:  0.3
    k_CT:  0.01
    k_q:  10.25 # Default value, cannot be learned

    # Number of steps to run
    num_steps: 200

    # Discard an initialisation period
    burn_in: 10

    # Time differential to use for the solver
    dt: 0.1

  # Section of training data to use as train and test data (default is entire training data)
  # Pass a python slice as an argument
  training_data_size: !slice [0, ~]
```

### Loading data
Instead of generating synthetic data, you can also load data from an `.h5` File.

```yaml
Data:
  load_from_dir: path/to/h5file
```

### Calibrating the Berlin dataset
The dataset of Covid figures for Berlin from February to October 2020 can be found at `data/Covid/Berlin_data/data.h5`,
and is calibrated in the `Berlin` configuration set:
```commandline
utopya run Covid --cs Berlin
```
It contains figures for the S, E, I, R, SY, C, H, D and an overal Q compartment (with subdivision).
For this reason, the loss function needs to be adjusted to drop the CT and D and combine the Q compartments.
Add the following entry to the `Training` entry of the configuration file (see the `cfgs/Berlin/run.yml`) file:
```yaml
Training:
  Berlin_data_loss: True
```
The data is sourced from the [MODUS Covid simulator](https://covid-sim.info/2020-11-12/secondLockdown). We thank the authors for their support in acquiring the data.

### Configuration sets

The following configuration sets are included in the model:

- `Static_parameters`: A toy example using static parameters. Data is generated synthetically
- `Dynamic_parameters`: A toy example using dynamic parameters. Data is generated synthetically
- `Berlin`: Calibrates the Berlin dataset
- `Berlin_MCMC`: runs a Langevin MCMC scheme on the Berlin dataset

You can run these sets simply by calling

```commandline
utopya run Covid --cs name_of_cs
```
