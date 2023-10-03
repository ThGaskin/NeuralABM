# The SIR model of contagious diseases

> **_Note_**: See the section on [Configuration sets](#configuration-sets) to see how to reproduce
> the plots from the [PNAS publication](https://www.pnas.org/doi/10.1073/pnas.2216415120).

### Model description
In the agent-based SIR model, ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DN)
agents move around a square domain ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5B0,%20L%5D%5E2)
with periodic boundary conditions, ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D0%20%3C%20L%20%5Cin%20%5Cmathbb%7BR%7D);
each agent has a position ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cmathbf%7Bx%7D_i),
and a state ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dk_i%20%5Cin%20%5C%7B%20%5Ctext%7BS,%20I,%20R%7D%20%5C%7D).
All agents with ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dk_i%20=%20%5Ctext%7BS%7D) are
susceptible to the disease. If a susceptible agent lies within the infection radius ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dr)
of an infected agent (an agent with ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dk_i%20=%20%5Ctext%7BI%7D)),
they are infected with infection probability ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dp).
After a certain recovery time ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ctau),
agents recover from the disease (upon which ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dk_i%20=%20%5Ctext%7BR%7D));
each agent's time since infection is stored in a state ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ctau_i).
Agents move randomly around the space with diffusivities ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Csigma_S),
![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Csigma_I),
and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Csigma_R).


Let ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DS(%5Cmathbf%7Bx%7D,%20t))
be the spatio-temporal distribution of susceptible agents (analogously ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DI)
and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DR)).
Assume we only observe the temporal densities
> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ctext%7BS%7D(t)%20=%20%5Cdfrac%7B1%7D%7BN%7D%5Cint_%5COmega%20S(%5Cmathbf%7Bx%7D,%20t)%20%5Cmathrm%7Bd%7D%5Cmathbf%7Bx%7D,)
>
applicable to the spread of an epidemic where we only see the counts of infected and recovered patients without any
location tracking or contact tracing. To these observations the neural network now fits the stochastic equations

> ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Cmathrm%7Bd%7D%5Ctext%7BS%7D%20=%20-%20%5Cbeta%20%5Ctext%7BSI%7D%20%5Cmathrm%7Bd%7Dt%20-%20%5Csigma%20%5Ctext%7BI%7D%20%5Ccirc%20%5Cmathrm%7Bd%7DW%20%20%5C%5C%20%20%20%5Cmathrm%7Bd%7D%5Ctext%7BI%7D%20=%20(%5Cbeta%20%5Ctext%7BS%7D%20-%20%5Ctau%5E%7B-1%7D)%5Ctext%7BI%7D%5Cmathrm%7Bd%7Dt%20&plus;%20%20%5Csigma%20%5Ctext%7BI%7D%20%5Ccirc%20%5Cmathrm%7Bd%7DW%20%5Cnonumber%20%5C%5C%20%20%20%5Cmathrm%7Bd%7D%5Ctext%7BR%7D%20=%20%5Ctau%5E%7B-1%7D%20%5Ctext%7BI%7D%5Cmathrm%7Bd%7Dt,)
>
where ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7DW) is a Wiener process,
and ![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7D%5Ccirc) represents the Stratonovich integral.

### Model parameters

```yaml
Data:
  synthetic_data:

    # How to generate synthetic data. Smooth densities can be generated directly using the SDE model,
    # or from an agent-based model
    type: from_ABM

    # Number of agents
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
```
The key `type` determines how training data is generated. Set it `from_ABM` to generate
densities from the ABM, or to `smooth` to generate smooth densities from the SDEs directly.

### Specifying the parameters to learn
You can learn any of the parameters
![equation](https://latex.codecogs.com/gif.image?%5Cinline%20%5Cdpi%7B110%7Dp,%20t,%20%5Csigma).
Specify which parameters to learn in the `Training` entry:
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
from this dataset. See the `Predictions` configuration set for an example.

### Configuration sets

The following configuration sets are included in the model:

- `ABM_data`: generates synthetic ABM data (fig. 2 in the publication)
- `Predictions` fits the SDE system to the ABM data and calculates the marginal densities on the
parameters (figs. 3 & 4 in the publication).

You can run these sets simply by calling

```commandline
utopya run SIR --cs name_of_cs
```
