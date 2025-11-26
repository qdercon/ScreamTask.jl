# ScreamTask.jl

A minimal Julia package to simulate and plot data from the latent cause inference model outlined by [Berwian *et al.* (2025)](https://osf.io/preprints/psyarxiv/2kdtf_v1) (*1*). 

**This package is in early development and so usage with participant data without independent validation is not recommended.**

## The task and model

This model was developed to capture behaviour in the scream task, a Pavlovian fear conditioning paradigm in which human participants learn to associate one of two conditioned stimuli (CS+; in this case a moon) with an aversive unconditioned stimulus (US; a scream sound), while the other conditioned stimulus (CS-; in this case, a candle) is never paired with the US.

The model is a latent cause inference model, which assumes that participants infer the presence of latent causes (or contexts) that generate the observed CS-US contingencies. The model uses Bayesian inference to update beliefs about these latent causes based on the observed data, allowing it to capture complex patterns of learning and memory, such as spontaneous recovery of fear after extinction. These beliefs are assumed to be reflected by participant ratings of US expectancy for each of the CSs; these ratings are given every few trials during the task.

Due to the dependence of updates on the entire history of observations (and hence a combinatoric explosion in the number of possible latent cause assignments after two or three trials), the model is not exactly tractable, and approximate inference methods are used to simulate behaviour. In particular, the model uses a particle filter approach to approximate the posterior distribution over latent causes and their associated parameters.

## Example simulated behaviour

Simulated behaviour in the scream task using the latent cause inference model. The plot shows US expectancy ratings (y-axis) across trials (x-axis) for both the CS+ (blue line) and CS- (orange line). The shaded areas represent the break periods, for the purposes of simulating what happens while participants are completing other parts of the experiment such as questionnaires.

![](examples/predicted_trajectory.svg)


## Installation and usage

To install the package, simply refer to the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/qdercon/ScreamTask.git")
```

A complete example usage script is provided in [`examples/scream_task.jl`](examples/scream_task.jl). The rough outline of how the package is intended to be used is as follows.

### 1. Simulate data from example participants

Here, we define parameter distributions for the model parameters, fixing the Chinese Restaurant Process concentration parameter α to a constant value of 0.03, and sampling the other parameters from appropriate prior distributions. We then simulate data for 10 participants. The arguments referring to spacing and intermittent ratings are used to simulate the real human data, where scream expectancy ratings were only collected intermittently during the task, and never during the two break periods.

Note that the task structure in terms of trials-per-phase is hard-coded, but can be modified in the source code file [`src/simulate.jl`](src/simulate.jl) if needed.

```julia
# using Base.Threads
# Threads.nthreads()

using Distributions
using ScreamTask

parameter_distributions = Dict(
    :α => 0.03, # following Berwian et al. (2025)
    :a_obs => truncated(Gamma(1.0, 1.0), 0.0, 10.0),
    :γ => Beta(4.0, 2.0),
    :ω => Beta(1.0, 1.0),
    :ε => Beta(1.0, 1.0)
);

simulations = simulate(parameter_distributions;
                       seed=123,
                       n_ppts=10,
                       intermittent=true,
                       rating_spacing=(2, 3),
                       sparse_ratings=true);

# plot an example participant
plot_scream(simulations.full;
            type="task",
            individual=true,
            participant_no=1,
            plot_size=(900, 400))
```

Note that, to use multi-threading, you will have to start Julia with multiple threads, e.g. ```julia -t 8```; the commented lines will then print how many threads are available.

### 2. Optimize model parameters to fit simulated data

If using real-life data, you would load it, ensuring it has the correct format as the simulations. The ```optimize!``` function can then be used to fit the model to the data. The fastest, and most consistent optimization method (```:PRIMA```) uses the BOBYQA algorithm as implemented in [```PRIMA.jl```](https://github.com/libprima/PRIMA.jl) package.

Two alternative optimizers are also available: ```:BBO``` picks a black-box optimizer from the [```BlackBoxOptim.jl```](https://github.com/robertfeldt/BlackBoxOptim.jl) package, while ```:NOMAD``` uses implements the mesh adaptive direct search algorithm [NOMAD](https://www.gerad.ca/en/software/nomad/), via its Julia interface [```NOMAD.jl```](https://github.com/bbopt/NOMAD.jl). Note that only ```:PRIMA``` and ```:BBO``` support multi-threading; ```:NOMAD``` runs serially so is much slower.

```julia
# initial values as the mode of the prior distributions
initial_parameters = Dict(
    :a_obs => 1.0,
    :γ => 0.6667,
    :ω => 0.5,
    :ε => 0.5
);

# bounds as per Berwian et al. (2025)
bounds = Dict(
    :a_obs => (0.0, 10.0),
    :γ => (0.0, 1.0),
    :ω => (0.0, 1.0),
    :ε => (0.0, 1.0)
);
free_params = collect(keys(bounds)); # for plotting
observed_data = simulations.observed;
true_parameters = simulations.parameters;

# perform optimization
results = optimize!(observed_data, bounds;
                    fixed_parameters=Dict(:α => 0.03),
                    optimizer=:PRIMA, # could also be :BBO or :NOMAD
                    n_particles=1_000,
                    resample_threshold=0.5,
                    initial_parameters=initial_parameters, # "modal"
                    start_perturbation=0.1,
                    use_threads=true);

plot_scream(results;
            type="recovery",
            true_parameters=true_parameters, 
            free_parameters=free_params,
            plot_size=(600, 900))
```
At least when using ```:PRIMA```, parameter recovery is highly dependent on the initial parameter values. As such, recovery of some parameters (particularly the selective maintenance parameter ω) can be poor initially. This could be improved by using a hierarchical fitting approach, but a simple solution appears to be to re-run optimization using the best-fitting parameters from an initial run as the new initial parameters. See the example script [`examples/scream_task.jl`](examples/scream_task.jl) for details.

## Next steps

- **Full benchmarking of different optimizers**: ```:PRIMA``` is definitely quickest, but ```:BBO``` and ```:NOMAD``` may be more robust in some cases.
- **Comparison to BADS**: In the original paper, Berwian *et al.* (2025) used the [BADS](https://github.com/acerbilab/bads) optimizer, which is not currently implemented in Julia (though does have a Python interface). This optimizer is fairly slow and computationally expensive (comparable to ```:NOMAD```), but shows robust parameter recovery in the original work. As such, it would be useful to compare the performance

## References

1. Berwian, I., Ren, Y., Pisupti, S., Ding, J., Moon, S., Chiu, J., Chandrasekhar, D., Niv, Y. (2025). Selective maintenance of aversive memories as a mechanism of spontaneous recovery of fear. Preprint on *PsyArXiv*. https://doi.org/10.31234/osf.io/2kdtf_v1