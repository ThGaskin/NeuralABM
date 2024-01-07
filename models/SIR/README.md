# The SIR model of contagious diseases

## Model description
In the agent-based SIR model, $N$ agents move around a square domain $[0, L]^2$, $0 < L \in \mathbb{R}$. Each agent $i$ has a position $\mathbf{x}_i$ and a state $k_i \in \{ S, I, R \}$. All agents with $k_i = S$ are susceptible to the disease. If a susceptible agent lies within the infection radius $r$ of an infected agent (an agent with $k_i = I$), they are infected with infection probability $\beta$. After a certain recovery time $\tau$, agents recover from the disease (upon which $k_i = R$);
each agent's time since infection is stored in a state $\tau_i$. Agents move randomly around the space with diffusivities $\sigma_S, \sigma_i, \sigma_R$:

<img src="https://github.com/ThGaskin/NeuralABM/assets/22022754/605b6e09-703e-4296-a884-ea1315dba8ea" width=60%>

Let $S(\mathbf{x},t)$ be the spatio-temporal distribution of susceptible agents (analogously $I(\mathbf{x},t)$ and $R(\mathbf{x},t)$ ). Assume we only observe the temporal densities $$S(t) = \dfrac{1}{N} \int_\Omega S(\mathbf{x}, t) \mathrm{d}\mathbf{x}$$ applicable to the spread of an epidemic where we only see the counts of infected and recovered patients without any
location tracking or contact tracing. To these observations the neural network now fits the stochastic equations $$\mathrm{d}S = - \beta S I \mathrm{d}t - \sigma I \mathrm{d}W$$ $$\mathrm{d}I = (\beta S - \tau^{-1} I) \mathrm{d}t + \sigma I \mathrm{d}W$$ $$\mathrm{d}R = \tau^{-1} I \mathrm{d}t$$ where $W$ is a Wiener process.

The parameters that can be calibrated are $\beta, \tau$, and $\sigma$. The model produces outputs such as a predicted time series for each compartment

<img src="https://github.com/ThGaskin/NeuralABM/files/13787421/densities_from_joint.pdf" width=100%>

as well as marginal densities on the parameters that are to be learned:

<img src="https://github.com/ThGaskin/NeuralABM/files/13787439/marginals.pdf" width=100%>

To ensure better training, the parameters can be scaled to ensure they are all of roughly equal magnitude when learned by the neural network.
The scaling factors can be controlled via the `Training.scaling_factors` dictionary. By default, the infection time is scaled by a factor of 10.

## Model parameters

```yaml
Data:
  synthetic_data:

    # How to generate synthetic data. Smooth densities can be generated directly using the SDE model,
    # or from an agent-based model
    type: from_ABM  # options: from_ABM or smooth

    # Number of agents (only relevant for the ABM)
    N: 150

    # Domain extent, and whether the boundaries are periodic
    space: [10, 10]
    is_periodic: false

    # Model parameters: infection radius, infection probability, infection time
    r_infectious: 1.0
    p_infect: 0.5
    t_infectious: 30

    # Agent diffusivities
    sigma_s: 0.15
    sigma_i: 0.03
    sigma_r: 0.15

    # Noise parameter for the smooth model
    sigma: 0.1

    # Number of steps to run
    num_steps: 200
Training:
  # Scale the infection time by a factor of 10 to ensure all learned parameters are of equal magnitude.
  # This makes training more efficient.
  scaling_factors:
    t_infectious: 10
```
The key `type` determines how training data is generated. Set it to `from_ABM` to generate
densities from the ABM, or to `smooth` to generate smooth densities from the SDEs directly.

### Specifying the parameters to learn
You can learn any subset of the parameters $\beta, \tau, \sigma$. Specify which parameters to learn in the `Training` entry:
```yaml
Training:
  to_learn: [p_infect, t_infectious, sigma]
```
For those parameters you do not wish to learn you must supply a true value:
```yaml
Training:
  to_learn: [p_infect, sigma]
  true_parameters:
    t_infectious: 14
```

### Loading data
Instead of generating synthetic data, you can also load data from an `.h5` File.

```yaml
Data:
  load_from_dir: path/to/h5file
```
This file must contain a densities dataset called `true_counts`. If you generate synthetic data
from the ABM first and then point the config to the location of the data, it will automatically run the model
from this dataset. See the `Predictions_on_ABM_data` configuration set for an example.

### Configuration sets
The default options generate SIR data from an ABM and calibrate it using a single neural training run.
You will notice the results are only moderately accurately. The ``Predictions_on_ABM_data`` configuration
set takes the same data but runs multiple chains -- you will notice a large jump in accuracy.

The following configuration sets are included in the model:

- `Generate_ABM_data` generates synthetic ABM data. This data is calibrated in the following configuration set:
- `Predictions_on_ABM_data` fits the SDE system to noisy ABM data and calculates the marginal densities on the
parameters (fig. 4 in the PNAS paper)
- `Generate_ground_truth` runs a two-dimensional grid search over two parameters to generate a ground truth distribution.
This data is calibrated in the following two configuration sets:
- `Predictions_on_smooth_data` fits the SDE system to noisy data generated by the same SDE and calculates the marginal
densities on the parameters (fig. 1d in the Epidemic forecasing paper)
- `MCMC_predictions_on_smooth_data`: runs an MCMC scheme on the same data as `Predictions_on_smooth_data` and plots the
results.

You can run these sets simply by calling

```commandline
utopya run SIR --cs <name_of_cs>
```
