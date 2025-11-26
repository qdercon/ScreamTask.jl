# Trial structure constants for the Scream Task
const N_TRIALS_ACQ = 28
const N_BREAK1 = 9
const N_TRIALS_EXT = 30
const N_BREAK2 = 34
const N_TRIALS_SR = 16
const N_TRIALS_RELEARN = 16
const TOTAL_TRIALS = N_TRIALS_ACQ + N_BREAK1 + N_TRIALS_EXT + N_BREAK2 + N_TRIALS_SR + N_TRIALS_RELEARN

const PHASE_BOUNDARIES = (
    acquisition = (1, N_TRIALS_ACQ),
    break1 = (N_TRIALS_ACQ + 1, N_TRIALS_ACQ + N_BREAK1),
    extinction = (N_TRIALS_ACQ + N_BREAK1 + 1, N_TRIALS_ACQ + N_BREAK1 + N_TRIALS_EXT),
    break2 = (N_TRIALS_ACQ + N_BREAK1 + N_TRIALS_EXT + 1,
              N_TRIALS_ACQ + N_BREAK1 + N_TRIALS_EXT + N_BREAK2),
    spontaneous_recovery = (N_TRIALS_ACQ + N_BREAK1 + N_TRIALS_EXT + N_BREAK2 + 1,
                            N_TRIALS_ACQ + N_BREAK1 + N_TRIALS_EXT + N_BREAK2 + N_TRIALS_SR),
    relearning = (TOTAL_TRIALS - N_TRIALS_RELEARN + 1, TOTAL_TRIALS)
)

function _generate_trial_schedule(rng::AbstractRNG)
    observations = zeros(Int, TOTAL_TRIALS)
    current_idx = 1

    acq_indices = 1:N_TRIALS_ACQ
    cs_plus_indices = shuffle!(rng, collect(acq_indices))[1:16]
    cs_minus_indices = setdiff(collect(acq_indices), cs_plus_indices)
    us_trials = shuffle!(rng, copy(cs_plus_indices))[1:8]
    no_us_trials = setdiff(cs_plus_indices, us_trials)
    observations[us_trials] .= 4
    observations[no_us_trials] .= 1
    observations[cs_minus_indices] .= 2
    current_idx += N_TRIALS_ACQ

    observations[current_idx:(current_idx + N_BREAK1 - 1)] .= 3
    current_idx += N_BREAK1

    ext_indices = current_idx:(current_idx + N_TRIALS_EXT - 1)
    cs_plus_indices = shuffle!(rng, collect(ext_indices))[1:18]
    cs_minus_indices = setdiff(collect(ext_indices), cs_plus_indices)
    observations[cs_plus_indices] .= 1
    observations[cs_minus_indices] .= 2
    current_idx += N_TRIALS_EXT

    observations[current_idx:(current_idx + N_BREAK2 - 1)] .= 3
    current_idx += N_BREAK2

    sr_indices = current_idx:(current_idx + N_TRIALS_SR - 1)
    for i in sr_indices
        observations[i] = rand(rng) < 0.5 ? 1 : 2
    end
    current_idx += N_TRIALS_SR

    relearn_indices = current_idx:(current_idx + N_TRIALS_RELEARN - 1)
    cs_plus_indices = shuffle!(rng, collect(relearn_indices))[1:10]
    cs_minus_indices = setdiff(collect(relearn_indices), cs_plus_indices)
    us_trials = cs_plus_indices[1:4]
    no_us_trials = cs_plus_indices[5:10]
    observations[us_trials] .= 4
    observations[no_us_trials] .= 1
    observations[cs_minus_indices] .= 2

    if !all(1 .<= observations .<= 4)
        invalid_idx = findall(x -> !(1 <= x <= 4), observations)
        error("Invalid trial schedule detected at indices $invalid_idx")
    end

    return observations
end

function _coerce_mask(values::AbstractVector, n_trials::Int)
    length(values) == n_trials || error("Provided rating_mask has length $(length(values)); expected $n_trials")
    mask = Vector{Bool}(undef, n_trials)
    for (idx, val) in enumerate(values)
        if val isa Bool
            mask[idx] = val
        elseif val isa Integer
            mask[idx] = val != 0
        elseif val isa AbstractFloat
            mask[idx] = val != 0.0
        else
            error("rating_mask values must be boolean-like, got $(typeof(val)) at position $(idx)")
        end
    end
    return mask
end

function _enforce_skip_types!(mask::Vector{Bool}, trial_types::Vector{Int}, skip_types::AbstractVector{Int})
    skip = Set(skip_types)
    for i in eachindex(mask)
        if trial_types[i] in skip
            mask[i] = false
        end
    end
    return mask
end

function _prepare_rating_mask(trial_types::Vector{Int},
                              rating_mask::Union{Nothing, AbstractVector},
                              intermittent::Bool,
                              rating_spacing::Tuple{Vararg{Int}},
                              rating_skip_types::AbstractVector{Int})
    if rating_mask !== nothing
        local_mask = _coerce_mask(rating_mask, length(trial_types))
    elseif intermittent
        local_mask = intermittent_rating_mask(trial_types; spacing=rating_spacing, skip_types=rating_skip_types)
    else
        local_mask = fill(true, length(trial_types))
    end
    return _enforce_skip_types!(local_mask, trial_types, rating_skip_types)
end

function _dense_from_sparse(n::Int, indices::Vector{Int}, values::Vector{Float64})
    dense = fill(NaN, n)
    for (idx, trial_idx) in enumerate(indices)
        dense[trial_idx] = values[idx]
    end
    return dense
end

function simulate(parameters::Dict;
                  seed::Int=42,
                  n_particles::Int=1_000,
                  resample_threshold::Float64=0.5,
                  compute_trajectories::Bool=true,
                  n_ppts::Int=1,
                  intermittent::Bool=true,
                  rating_spacing::Tuple{Vararg{Int}}=(2, 3),
                  rating_mask::Union{Nothing, AbstractVector}=nothing,
                  rating_skip_types::AbstractVector{Int}=Int[3],
                  sparse_ratings::Bool=true,
                  use_threads::Bool=true)
    n_ppts > 0 || error("n_ppts must be positive")
    master_rng = Random.MersenneTwister(seed)
    store_trajectories = n_ppts == 1 ? compute_trajectories : false

    full_data = Vector{Dict{Symbol, Any}}(undef, n_ppts)
    observed_data = Vector{Dict{Symbol, Any}}(undef, n_ppts)
    parameter_values = Vector{Dict{Symbol, Float64}}(undef, n_ppts)

    ppt_rngs = [Random.MersenneTwister(rand(master_rng, UInt)) for _ in 1:n_ppts]
    run_threaded = use_threads && n_ppts > 1 && Base.Threads.nthreads() > 1
    progress = Atomic{Int}(0)
    total = n_ppts

    function simulate_single(ppt::Int)
        ppt_rng = ppt_rngs[ppt]
        trial_types = _generate_trial_schedule(ppt_rng)
        alpha, a_obs, decay, omega, epsilon = resolve_model_parameters(parameters; randomize_missing=true, rng=ppt_rng)
        param_dict = Dict(:α => alpha, :a_obs => a_obs, :γ => decay, :ω => omega, :ε => epsilon)
        parameter_values[ppt] = copy(param_dict)

        mask = _prepare_rating_mask(trial_types, rating_mask, intermittent, rating_spacing, rating_skip_types)
        pf_seed = seed + ppt - 1
        pf_result = run_particle_filter(trial_types, alpha, a_obs, decay, omega, epsilon;
                                        n_particles=n_particles,
                                        resample_threshold=resample_threshold,
                                        seed=pf_seed,
                                        compute_trajectories=store_trajectories,
                                        intermittent=false,
                                        rating_mask=mask,
                                        rating_skip_types=rating_skip_types)

        rating_indices = pf_result["rating_indices"]
        rating_values = pf_result["trial_mean"][rating_indices]
        dense_ratings = _dense_from_sparse(length(trial_types), rating_indices, rating_values)

        full_entry = Dict{Symbol, Any}(
            :participant => ppt,
            :trial_types => trial_types,
            :trial_predictions => pf_result["trial_mean"],
            :trial_sd => pf_result["trial_sd"],
            :cs_plus_mean => pf_result["cs_plus_mean"],
            :cs_plus_sd => pf_result["cs_plus_sd"],
            :cs_minus_mean => pf_result["cs_minus_mean"],
            :cs_minus_sd => pf_result["cs_minus_sd"],
            :rating_mask => mask,
            :rating_indices => rating_indices,
            :parameters => copy(param_dict),
            :particle_filter => store_trajectories ? pf_result : nothing,
            :phase_bounds => PHASE_BOUNDARIES
        )
        full_data[ppt] = full_entry

        observed_entry = Dict{Symbol, Any}(
            :participant => ppt,
            :trial_types => trial_types,
            :rating_trials => rating_indices,
            :rating_types => trial_types[rating_indices],
            :ratings => sparse_ratings ? rating_values : dense_ratings,
            :ratings_sparse => rating_values,
            :ratings_full => dense_ratings,
            :mask => mask,
            :sparse => sparse_ratings,
            :true_parameters => copy(param_dict)
        )
        observed_data[ppt] = observed_entry

        return ppt
    end

    @withprogress name = "simulating" begin
        if run_threaded
            Base.Threads.@threads for ppt in 1:n_ppts
                participant = simulate_single(ppt)
                done = atomic_add!(progress, 1) + 1
                frac = min(done / total, 1.0)
                @logprogress "simulating datasets" frac done=done total=total participant=participant
            end
        else
            for ppt in 1:n_ppts
                participant = simulate_single(ppt)
                done = atomic_add!(progress, 1) + 1
                frac = min(done / total, 1.0)
                @logprogress "simulating datasets" frac done=done total=total participant=participant
            end
        end
    end

    return (full=full_data, observed=observed_data, parameters=parameter_values)
end

function simulate_lci_model(trial_types::Vector{Int},
                            alpha::Float64=1.0,
                            a_obs::Float64=1.0,
                            gamma::Float64=1.0,
                            omega::Float64=0.0,
                            epsilon::Float64=0.0,
                            rng::AbstractRNG=Random.GLOBAL_RNG)
    n_trials = length(trial_types)
    n_types = 4
    max_causes = n_trials
    causes = zeros(Int, n_trials)
    counts = zeros(Int, max_causes)
    obs_counts = zeros(Int, max_causes, n_types)
    predictions = zeros(Float64, n_trials)
    predictions_cs_plus = zeros(Float64, n_trials)
    predictions_cs_minus = zeros(Float64, n_trials)
    us_probs = zeros(Float64, max_causes)

    local_effective_count(t::Int, cause::Int; obs_type::Union{Nothing, Int}=nothing) =
        effective_count(causes, counts, obs_counts, us_probs,
                        t, cause, gamma, omega, trial_types;
                        obs_type=obs_type, epsilon=epsilon)

    cs_plus_stim_types = Int[1, 4]
    cs_plus_us_types = Int[4]
    cs_minus_stim_types = Int[2]
    cs_minus_us_types = Int[]

    K = 0
    for t in 1:n_trials
        probs = zeros(Float64, K + 1)
        for k in 1:K
            eff_count = local_effective_count(t, k)
            probs[k] = eff_count / (t - 1 + alpha)
            obs_count = Float64(obs_counts[k, trial_types[t]])
            a_post = a_obs + obs_count
            b_post = a_obs + max(eff_count - obs_count, 0.0)
            phi = (a_post + b_post) == 0.0 ? 0.0 : a_post / (a_post + b_post)
            probs[k] *= phi
        end
        probs[K + 1] = alpha / (t - 1 + alpha)
        probs[K + 1] *= 0.5
        cause = sample(rng, 1:(K + 1), Weights(probs))
        causes[t] = cause
        if cause > K
            K += 1
        end
        counts[cause] += 1
        obs_counts[cause, trial_types[t]] += 1
        cs_us_count = Float64(obs_counts[cause, 4])
        total_count = Float64(counts[cause])
        a_us = a_obs + cs_us_count
        b_us = a_obs + total_count - cs_us_count
        us_probs[cause] = (a_us + b_us) == 0.0 ? 0.5 : a_us / (a_us + b_us)
        predictions[t] = us_probs[cause]
        get_eff_count(cause_idx::Int, obs_type::Union{Nothing, Int}=nothing) =
            local_effective_count(t + 1, cause_idx; obs_type=obs_type)
        trials_seen = t
        predictions_cs_plus[t] = rating_prediction(get_eff_count, K, cs_plus_stim_types, cs_plus_us_types,
                                                   trials_seen, alpha, a_obs, n_types)
        predictions_cs_minus[t] = rating_prediction(get_eff_count, K, cs_minus_stim_types, cs_minus_us_types,
                                                    trials_seen, alpha, a_obs, n_types)
    end

    return Dict(
        "causes" => causes,
        "predictions" => predictions,
        "cs_plus_predictions" => predictions_cs_plus,
        "cs_minus_predictions" => predictions_cs_minus,
        "us_probabilities" => us_probs[1:K],
        "n_causes" => K
    )
end
