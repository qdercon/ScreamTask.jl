using Distributions
using ScreamTask

# using Base.Threads
# Threads.nthreads()

# --- Simulate the scream task -------------------------------------------------

parameter_distributions = Dict(
    :α => 0.03, # truncated(Gamma(1.0, 1.0), 0.0, 10.0),
    :a_obs => truncated(Gamma(1.0, 1.0), 0.0, 10.0),
    :γ => Beta(4.0, 2.0),
    :ω => Beta(1.0, 1.0),
    :ε => Beta(1.0, 1.0)
);

simulations = simulate(parameter_distributions;
                       seed=123,
                       n_ppts=50,
                       intermittent=true,
                       rating_spacing=(2, 3), # ratings every 2 or 3 trials
                       sparse_ratings=true);

# --- Plot simulated data -------------------------------------------------------

full_data = simulations.full;
plot_scream(full_data; type="task", individual=true, participant_no=1, plot_size=(1000, 800))
plot_scream(full_data; type="task", individual=false, plot_size=(900, 400))

# --- Parameter recovery via optimization --------------------------------------

# initial values as the mode of the prior distributions
initial_parameters = Dict(
    :a_obs => 1.0,
    :γ => 0.6667,
    :ω => 0.5,
    :ε => 0.5
);

bounds = Dict(
    :a_obs => (0.0, 10.0),
    :γ => (0.0, 1.0),
    :ω => (0.0, 1.0),
    :ε => (0.0, 1.0)
);
free_params = collect(keys(bounds));

observed_data = simulations.observed;
true_parameters = simulations.true_parameters;
results = optimize!(observed_data, bounds;
                    fixed_parameters=Dict(:α => 0.03),
                    optimizer=:PRIMA, # could also be :BBO or :NOMAD
                    n_particles=1_000,
                    resample_threshold=0.5,
                    initial_parameters=initial_parameters, # "modal"
                    start_perturbation=0.1,
                    use_threads=true);

# get estimates for each participant to use as initial values for re-optimization
estimates = [res.true_parameters for res in results];
res_reopt = optimize!(observed_data, bounds;
                    fixed_parameters=Dict(:α => 0.03),
                    optimizer=:PRIMA,
                    n_particles=1_000,
                    resample_threshold=0.5,
                    initial_parameters=estimates, # "modal"
                    start_perturbation=0.1,
                    use_threads=true);

# --- Plot parameter recovery results ------------------------------------------

plot_scream(
    results; type="recovery", true_parameters=true_parameters, 
    free_parameters=free_params, plot_size=(600, 900)
)
plot_scream(
    res_reopt; type="recovery", true_parameters=true_parameters, 
    free_parameters=free_params, plot_size=(600, 900)
)