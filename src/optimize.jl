function _has_dataset_key(data, key::Symbol)
    if data isa AbstractDict
        return haskey(data, key)
    elseif data isa NamedTuple
        return hasproperty(data, key)
    else
        return hasproperty(data, key)
    end
end

function _dataset_value(data, key::Symbol)
    if data isa AbstractDict
        haskey(data, key) || error("Dataset entry is missing key $(key).")
        return data[key]
    elseif hasproperty(data, key)
        return getproperty(data, key)
    else
        error("Dataset entry of type $(typeof(data)) is missing field $(key).")
    end
end

function _normalize_dataset_entry(data)
    trial_types = Int.(_dataset_value(data, :trial_types))
    participant = _has_dataset_key(data, :participant) ? _dataset_value(data, :participant) : nothing

    mask = if _has_dataset_key(data, :mask)
        raw_mask = _dataset_value(data, :mask)
        raw_mask isa Vector{Bool} ? copy(raw_mask) : _coerce_mask(raw_mask, length(trial_types))
    else
        nothing
    end
    if mask !== nothing
        _enforce_skip_types!(mask, trial_types, Int[3])
    end

    rating_trials = if _has_dataset_key(data, :rating_trials)
        Int.(_dataset_value(data, :rating_trials))
    elseif mask !== nothing
        findall(mask)
    else
        Int[]
    end

    raw_ratings = _dataset_value(data, :ratings)
    ratings_vec = Float64.(raw_ratings)

    if isempty(rating_trials)
        if length(ratings_vec) == length(trial_types)
            rating_trials = findall(x -> isfinite(x), ratings_vec)
        else
            error("Dataset entry $(participant === nothing ? "" : string(participant)) must provide either :rating_trials, :mask, or a full-length ratings vector.")
        end
    end

    rating_trials = sort(unique(rating_trials))
    rating_trials = [idx for idx in rating_trials if 1 <= idx <= length(trial_types)]

    rating_values = if length(ratings_vec) == length(rating_trials)
        ratings_vec
    elseif length(ratings_vec) == length(trial_types)
        ratings_vec[rating_trials]
    elseif _has_dataset_key(data, :ratings_sparse)
        Float64.(_dataset_value(data, :ratings_sparse))
    else
        error("ratings vector must match either the number of rating trials or total trials.")
    end

    true_params = _has_dataset_key(data, :true_parameters) ? Dict{Symbol, Float64}(_dataset_value(data, :true_parameters)) : nothing

    # Ensure we never treat break stimuli as observed ratings
    valid_mask = [trial_types[idx] != 3 for idx in rating_trials]
    rating_trials = rating_trials[valid_mask]
    rating_values = rating_values[valid_mask]

    return (trial_types=trial_types,
            rating_trials=rating_trials,
            ratings=Float64.(rating_values),
            participant=participant,
            mask=mask,
            true_parameters=true_params)
end

function _rating_mask_from_trials(n_trials::Int, rating_trials::Vector{Int})
    mask = fill(false, n_trials)
    for idx in rating_trials
        1 <= idx <= n_trials || error("Rating trial index $(idx) is out of bounds for $(n_trials) trials.")
        mask[idx] = true
    end
    return mask
end

function _estimate_rating_sigma(mu_vals::Vector{Float64}, obs_vals::Vector{Float64})
    n = length(mu_vals)
    if n <= 1
        return MIN_RATING_SIGMA
    end
    diff = obs_vals .- mu_vals
    variance = sum(diff .^ 2) / max(n - 1, 1)
    if !isfinite(variance) || variance < 0
        return MIN_RATING_SIGMA
    end
    sigma = sqrt(variance)
    return isfinite(sigma) ? max(sigma, MIN_RATING_SIGMA) : MIN_RATING_SIGMA
end

function _build_particle_objective(trial_types::Vector{Int},
                                   rating_trials::Vector{Int},
                                   ratings::Vector{Float64};
                                   parameter_keys::Vector{Symbol},
                                   fixed_parameters::Dict{Symbol, Float64},
                                   n_particles::Int,
                                   resample_threshold::Float64,
                                   seed::Int,
                                   mask::Union{Nothing, Vector{Bool}})
    isempty(parameter_keys) && error("At least one free parameter is required for optimisation.")
    length(ratings) == length(rating_trials) || error("ratings vector must align with rating_trials.")

    local_mask = mask === nothing ? _rating_mask_from_trials(length(trial_types), rating_trials) : copy(mask)

    valid_pairs = [(trial_idx, rating) for (trial_idx, rating) in zip(rating_trials, ratings) if isfinite(rating)]
    isempty(valid_pairs) && error("No valid rating observations were provided.")
    clean_trials = [pair[1] for pair in valid_pairs]
    clean_ratings = [pair[2] for pair in valid_pairs]

    function objective(θ::AbstractVector{<:Real})
        length(θ) == length(parameter_keys) || throw(ArgumentError("Expected $(length(parameter_keys)) parameters, got $(length(θ))"))
        params = Dict{Symbol, Any}(fixed_parameters)
        for (key, value) in zip(parameter_keys, θ)
            params[key] = Float64(value)
        end

        alpha, a_obs, decay, omega, epsilon = resolve_model_parameters(params; randomize_missing=false)
        pf_result = run_particle_filter(trial_types, alpha, a_obs, decay, omega, epsilon;
                                        n_particles=n_particles,
                                        resample_threshold=resample_threshold,
                                        seed=seed,
                                        compute_trajectories=false,
                                        intermittent=false,
                                        rating_mask=local_mask)

        preds = pf_result["trial_mean"][clean_trials]
        mu_vals = clamp.(Float64.(preds), 0.0, 1.0)
        obs_vals = clamp.(Float64.(clean_ratings), 0.0, 1.0)
        sigma = _estimate_rating_sigma(mu_vals, obs_vals)

        ll = 0.0
        for (raw, obs, mu) in zip(clean_ratings, obs_vals, mu_vals)
            ll += normal_rating_logprob(Float64(raw), obs, mu, sigma)
        end
        return -ll
    end

    return objective
end

function _init_start_vector(free_keys::Vector{Symbol},
                            bounds::Dict{Symbol, Tuple{Float64, Float64}},
                            initial_parameters::Union{Nothing, Dict{Symbol, Float64}},
                            start_perturbation::Float64,
                            rng::AbstractRNG)
    start = Vector{Float64}(undef, length(free_keys))
    for (idx, key) in enumerate(free_keys)
        lower, upper = bounds[key]
        base_val = initial_parameters === nothing ? (lower + upper) / 2 : get(initial_parameters, key, (lower + upper) / 2)
        perturb = start_perturbation <= 0 ? 0.0 : rand(rng, Uniform(-start_perturbation, start_perturbation))
        guess = base_val * (1.0 + perturb)
        start[idx] = clamp(guess, lower, upper)
    end
    return start
end

function optimize!(datasets,
                   bounds::Dict{Symbol, Tuple{Float64, Float64}};
                   optimizer::Symbol=:PRIMA,
                   n_particles::Int=1_000,
                   resample_threshold::Float64=0.5,
                   rhobeg::Float64=0.05,
                   initial_parameters::Union{Nothing, Dict{Symbol, Float64}, Vector{Dict{Symbol, Float64}}}=nothing,
                   fixed_parameters::Union{Dict{Symbol, Float64}, Vector{Dict{Symbol, Float64}}}=Dict{Symbol, Float64}(),
                   start_perturbation::Float64=0.1,
                   seed::Int=0,
                   use_threads::Bool=true)
    data_entries = datasets isa AbstractVector ? collect(datasets) : [datasets]
    isempty(data_entries) && error("At least one dataset is required for optimisation.")

    free_keys = collect(keys(bounds))
    isempty(free_keys) && error("bounds must contain at least one parameter to estimate.")

    lower_bounds = [bounds[k][1] for k in free_keys]
    upper_bounds = [bounds[k][2] for k in free_keys]

    optimizer == :PRIMA || optimizer == :BBO || optimizer == :NOMAD || error("Unsupported optimizer $(optimizer). Choose :PRIMA, :BBO, or :NOMAD.")
    run_threaded = use_threads && length(data_entries) > 1 && Base.Threads.nthreads() > 1 && optimizer != :NOMAD

    results = Vector{NamedTuple{(:participant, :estimate, :optimizer, :true_parameters),
                                Tuple{Any, Dict{Symbol, Float64}, Symbol, Union{Nothing, Dict{Symbol, Float64}}}}}(undef, length(data_entries))

    progress = Atomic{Int}(0)
    total = length(data_entries)

    function run_single(idx::Int)
        entry = _normalize_dataset_entry(data_entries[idx])
        dataset_seed = seed + idx
        fixed = if fixed_parameters isa Vector
            length(fixed_parameters) >= idx || error("fixed_parameters vector must have at least $(length(data_entries)) entries.")
            fixed_parameters[idx]
        else
            fixed_parameters
        end
        objective = _build_particle_objective(entry.trial_types,
                                              entry.rating_trials,
                                              entry.ratings;
                                              parameter_keys=free_keys,
                                              fixed_parameters=fixed,
                                              n_particles=n_particles,
                                              resample_threshold=resample_threshold,
                                              seed=dataset_seed,
                                              mask=entry.mask)
        rng = Random.MersenneTwister(dataset_seed)
        init_params = if initial_parameters isa Vector
            length(initial_parameters) >= idx || error("initial_parameters vector must have at least $(length(data_entries)) entries.")
            initial_parameters[idx]
        else
            initial_parameters
        end
        start_guess = _init_start_vector(free_keys, bounds, init_params, start_perturbation, rng)

        if optimizer == :PRIMA
            best, _ = prima(objective, start_guess; xl=lower_bounds, xu=upper_bounds, rhobeg=rhobeg)
        elseif optimizer == :BBO
            sr = [(lower_bounds[i], upper_bounds[i]) for i in eachindex(free_keys)]
            res = bboptimize(objective; SearchRange = sr, NumDimensions = length(free_keys), MaxFuncEvals = 1000, TraceMode = :silent)
            best = best_candidate(res)
        else
            function nomad_eval(x)
                obj_val = objective(x)
                return (true, true, [obj_val])
            end

            pb = NOMAD.NomadProblem(
                length(free_keys),
                1,
                ["OBJ"],
                nomad_eval;
                lower_bound = lower_bounds,
                upper_bound = upper_bounds
            )

            pb.options.max_bb_eval = 1000
            pb.options.display_degree = 0

            res = NOMAD.solve(pb, start_guess)
            best = res.x_best_feas === nothing ? res.x_best_bbo : res.x_best_feas
            best === nothing && error("NOMAD did not return a feasible or best-so-far solution.")
        end
        estimate = Dict{Symbol, Float64}(fixed)
        for (key, val) in zip(free_keys, best)
            estimate[key] = val
        end
        if data_entries[idx] isa AbstractDict
            data_entries[idx][:estimated_parameters] = estimate
        end
        results[idx] = (participant=entry.participant,
                        estimate=estimate,
                        optimizer=optimizer,
                        true_parameters=entry.true_parameters)

        return entry.participant
    end

    @withprogress name = "optimising" begin
        if run_threaded
            Base.Threads.@threads for idx in eachindex(data_entries)
                participant = run_single(idx)
                done = atomic_add!(progress, 1) + 1
                frac = min(done / total, 1.0)
                @logprogress "optimising $(optimizer)" frac done=done total=total participant=participant
            end
        else
            for idx in eachindex(data_entries)
                participant = run_single(idx)
                done = atomic_add!(progress, 1) + 1
                frac = min(done / total, 1.0)
                @logprogress "optimising $(optimizer)" frac done=done total=total participant=participant
            end
        end
    end

    return results
end
