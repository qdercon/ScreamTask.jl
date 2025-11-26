mutable struct ParticleState
    causes::Vector{Int}
    highest_us_cause::Vector{Int}
    counts::Vector{Int}
    obs_counts::Matrix{Int}
    us_probs::Vector{Float64}
    n_causes::Int
    log_weight::Float64
end

const SMALL_PROBABILITY = 1.0e-12
const IN_RATING_SIGMA = 1.0e-3
const MIN_RATING_SIGMA = IN_RATING_SIGMA
const RATING_BIN_WIDTH = 0.01

using LogExpFunctions: logsumexp

function init_particle(n_trials::Int, n_types::Int)
    ParticleState(Int[], Int[], zeros(Int, n_trials), zeros(Int, n_trials, n_types), fill(0.5, n_trials), 0, 0.0)
end

function copy_particle(p::ParticleState)
    ParticleState(
        copy(p.causes), copy(p.highest_us_cause), copy(p.counts), copy(p.obs_counts), copy(p.us_probs),
        p.n_causes, p.log_weight
    )
end

@inline function make_rng(seed::Int)
    return seed == 0 ? Random.TaskLocalRNG() : MersenneTwister(seed)
end

function effective_count(causes::AbstractVector{<:Integer},
                         counts::AbstractVector{<:Real},
                         obs_counts::AbstractMatrix{<:Real},
                         us_probs::AbstractVector{<:Real},
                         t::Int, cause::Int,
                         decay::Float64,
                         omega::Float64,
                         trial_types::AbstractVector{<:Integer};
                         obs_type::Union{Nothing, Int}=nothing,
                         epsilon::Float64=0.0)
    if obs_type === nothing
        if decay == 1.0 && omega == 0.0 && epsilon == 1.0
            return Float64(counts[cause])
        end
    elseif decay == 1.0 && omega == 0.0 && epsilon == 1.0
        return Float64(obs_counts[cause, obs_type])
    end

    limit = min(length(causes), t - 1)
    eff_count = 0.0
    for i in 1:limit
        if causes[i] == cause && (obs_type === nothing || trial_types[i] == obs_type)
            
            # Determine the salience for trial `i`
            salience = trial_types[i] == 4 ? epsilon : 1.0

            # Calculate the decay factor for trial `i`
            decay_factor = 1.0
            if omega == 0.0
                decay_factor = decay^(t - i)
            else
                decay_prod = 1.0
                for _ in (i + 1):t
                    decay_prod *= decay + (1 - decay) * omega * Float64(us_probs[cause])
                end
                decay_factor = decay_prod
            end

            # Add the decayed and salience-weighted contribution of trial `i`
            eff_count += decay_factor * salience
        end
    end

    return eff_count
end

# ParticleState wrapper for effective_count
function effective_count(p::ParticleState,
                         t::Int, cause::Int,
                         decay::Float64,
                         omega::Float64,
                         trial_types::Vector{Int};
                         obs_type::Union{Nothing, Int}=nothing,
                         epsilon::Float64=0.0)
    return effective_count(p.causes, p.counts, p.obs_counts, p.us_probs,
                           t, cause, decay, omega, trial_types;
                           obs_type=obs_type, epsilon=epsilon)
end

function rating_prediction(get_eff_count::Function,
                           n_causes::Int,
                           stim_obs_types::AbstractVector{<:Integer},
                           us_obs_types::AbstractVector{<:Integer},
                           trials_seen::Int,
                           alpha::Float64,
                           a_obs::Float64,
                           n_types::Int)
    denom_base = trials_seen + alpha
    total_weight = 0.0
    us_weighted = 0.0

    for k in 1:n_causes
        eff_count = get_eff_count(k)
        if eff_count <= 0.0
            continue
        end

        prior_weight = denom_base == 0.0 ? 0.0 : eff_count / denom_base
        denom = eff_count + n_types * a_obs
        stim_eff = 0.0
        for obs_type in stim_obs_types
            stim_eff += get_eff_count(k, obs_type)
        end
        stim_prior = length(stim_obs_types) * a_obs
        stim_likelihood = denom <= 0.0 ? 0.0 : (stim_eff + stim_prior) / denom
        weight = prior_weight * stim_likelihood
        if weight <= 0.0
            continue
        end

        us_eff = 0.0
        for obs_type in us_obs_types
            us_eff += get_eff_count(k, obs_type)
        end
        no_us_eff = max(stim_eff - us_eff, 0.0)
        us_pred_cause = (us_eff + a_obs) / (us_eff + no_us_eff + 2 * a_obs)

        total_weight += weight
        us_weighted += weight * us_pred_cause
    end

    stim_count = length(stim_obs_types)
    new_cause_weight = denom_base == 0.0 ? stim_count / n_types : (alpha / denom_base) * (stim_count / n_types)
    total_weight += new_cause_weight
    us_weighted += new_cause_weight * 0.5

    return total_weight == 0.0 ? 0.5 : us_weighted / total_weight
end

@inline function normal_rating_logprob(obs_raw::Float64,
                                       obs_clamped::Float64,
                                       mu::Float64,
                                       sigma::Float64)
    dist = Normal(mu, sigma)
    prob = RATING_BIN_WIDTH * pdf(dist, obs_clamped)
    if obs_raw <= 0.0
        prob += cdf(dist, 0.0)
    elseif obs_raw >= 1.0
        prob += 1.0 - cdf(dist, 1.0)
    end
    return log(max(prob, SMALL_PROBABILITY))
end

# ParticleState wrapper for rating_prediction
function particle_rating_prediction(p::ParticleState,
                                    stim_obs_types::Vector{Int},
                                    us_obs_types::Vector{Int},
                                    t::Int,
                                    alpha::Float64,
                                    a_obs::Float64,
                                    decay::Float64,
                                    omega::Float64,
                                    epsilon::Float64,
                                    n_types::Int,
                                    trial_types::Vector{Int})
    get_eff_count(cause::Int, obs_type::Union{Nothing, Int}=nothing) =
        effective_count(p, t, cause, decay, omega, trial_types; obs_type=obs_type, epsilon=epsilon)
    trials_seen = t - 1
    return rating_prediction(get_eff_count, p.n_causes, stim_obs_types, us_obs_types,
                              trials_seen, alpha, a_obs, n_types)
end

function intermittent_rating_mask(trial_types::Vector{Int};
                                  spacing::Tuple{Vararg{Int}}=(2, 3),
                                  skip_types::AbstractVector{Int}=Int[3])
    n_trials = length(trial_types)
    length(spacing) == 0 && error("spacing must contain at least one interval")
    mask = fill(false, n_trials)
    spacing_seq = collect(spacing)
    any(x -> x <= 0, spacing_seq) && error("spacing values must be positive")
    spacing_idx = 1
    counter = 0
    for t in 1:n_trials
        trial_type = trial_types[t]
        if trial_type in skip_types
            continue
        end
        counter += 1
        if counter >= spacing_seq[spacing_idx]
            mask[t] = true
            counter = 0
            spacing_idx = spacing_idx == length(spacing_seq) ? 1 : spacing_idx + 1
        end
    end
    return mask
end

function weighted_stats(values::Vector{Float64}, weights::Vector{Float64})
    total_weight = sum(weights)
    if total_weight <= 0.0
        return 0.5, 0.0
    end
    norm_weights = weights ./ total_weight
    mean_value = sum(norm_weights .* values)
    var_value = sum(norm_weights .* (values .- mean_value).^2)
    return mean_value, sqrt(max(var_value, 0.0))
end

function normalize_weights(particles::Vector{ParticleState})
    log_weights = [p.log_weight for p in particles]
    max_log = maximum(log_weights)
    shifted = exp.(log_weights .- max_log)
    total = sum(shifted)
    if total <= 0.0
        n = length(particles)
        return fill(1.0 / n, n)
    end
    return shifted ./ total
end

function resample_particles(particles::Vector{ParticleState},
                            weights::Vector{Float64},
                            rng::AbstractRNG)
    n = length(particles)
    indices = sample(rng, 1:n, Weights(weights), n)
    new_particles = Vector{ParticleState}(undef, n)
    for (i, idx) in enumerate(indices)
        new_particles[i] = copy_particle(particles[idx])
        new_particles[i].log_weight = 0.0
    end
    new_weights = fill(1.0 / n, n)
    return new_particles, new_weights
end

function propagate_particle!(p::ParticleState,
                             t::Int,
                             trial_type::Int,
                             alpha::Float64,
                             a_obs::Float64,
                             decay::Float64,
                             omega::Float64,
                             epsilon::Float64,
                             n_types::Int,
                             trial_types::Vector{Int},
                             rng::AbstractRNG)
    denom = (t - 1) + alpha
    probs = zeros(Float64, p.n_causes + 1)

    for k in 1:p.n_causes
        eff_count = effective_count(p, t, k, decay, omega, trial_types)
        if eff_count <= 0.0
            continue
        end
        prior_weight = denom == 0.0 ? 0.0 : eff_count / denom
        denom_likelihood = eff_count + n_types * a_obs
        obs_eff = effective_count(p, t, k, decay, omega, trial_types; obs_type=trial_type, epsilon=epsilon)
        phi = denom_likelihood <= 0.0 ? 0.0 : (obs_eff + a_obs) / denom_likelihood
        probs[k] = max(prior_weight * phi, 0.0)
    end

    new_cause_prior = denom == 0.0 ? 1.0 : alpha / denom
    probs[p.n_causes + 1] = new_cause_prior * (1.0 / n_types)

    for i in eachindex(probs)
        if !isfinite(probs[i]) || probs[i] < 0.0
            probs[i] = 0.0
        end
    end

    pred_prob = sum(probs)
    pred_prob = isfinite(pred_prob) ? pred_prob : 0.0
    pred_prob = max(pred_prob, SMALL_PROBABILITY)
    p.log_weight += log(pred_prob)

    if pred_prob == 0.0
        cause_idx = p.n_causes + 1
    else
        normalized = probs ./ pred_prob

        # Guard against numerical issues that yield NaN/Inf weights
        total = 0.0
        for i in eachindex(normalized)
            w = normalized[i]
            if !isfinite(w) || w < 0.0
                normalized[i] = 0.0
            else
                total += w
            end
        end

        if total <= 0.0
            cause_idx = p.n_causes + 1
        else
            normalized ./= total
            cause_idx = sample(rng, 1:length(normalized), Weights(normalized))
        end
    end

    if cause_idx == p.n_causes + 1
        p.n_causes += 1
        cause_real = p.n_causes
        p.counts[cause_real] = 0
        p.obs_counts[cause_real, :] .= 0
        p.us_probs[cause_real] = 0.5
    else
        cause_real = cause_idx
    end

    push!(p.causes, cause_real)
    p.counts[cause_real] += 1
    p.obs_counts[cause_real, trial_type] += 1

    cs_us_count = effective_count(p, t + 1, cause_real, decay, omega, trial_types; obs_type=4, epsilon=epsilon)
    total_count = effective_count(p, t + 1, cause_real, decay, omega, trial_types)
    a_us = a_obs + cs_us_count
    b_us = a_obs + total_count - cs_us_count
    denom_us = a_us + b_us
    p.us_probs[cause_real] = denom_us <= 0.0 ? 0.5 : a_us / denom_us

    if p.n_causes == 0
        push!(p.highest_us_cause, 0)
    else
        _, best_idx = findmax(p.us_probs[1:p.n_causes])
        push!(p.highest_us_cause, best_idx)
    end
end

function run_particle_filter(trial_types::Vector{Int},
                             alpha::Float64,
                             a_obs::Float64,
                             decay::Float64,
                             omega::Float64,
                             epsilon::Float64;
                             n_particles::Int=1_000,
                             resample_threshold::Float64=0.5,
                             seed::Int=0,
                             compute_trajectories::Bool=false,
                             intermittent::Bool=false,
                             rating_spacing::Tuple{Vararg{Int}}=(2, 3),
                             rating_mask::Union{Nothing, AbstractVector}=nothing,
                             rating_skip_types::AbstractVector{Int}=Int[3])
    rng = make_rng(seed)
    n_trials = length(trial_types)
    n_types = 4

    local_rating_mask = if rating_mask !== nothing
        length(rating_mask) == n_trials || error("Provided rating_mask has length $(length(rating_mask)), expected $(n_trials)")
        tmp = Vector{Bool}(undef, n_trials)
        for (idx, val) in enumerate(rating_mask)
            if val isa Bool
                tmp[idx] = val
            elseif val isa Integer
                tmp[idx] = val != 0
            elseif val isa AbstractFloat
                tmp[idx] = val != 0.0
            else
                error("rating_mask values must be boolean, got $(typeof(val))")
            end
        end
        tmp
    elseif intermittent
        intermittent_rating_mask(trial_types; spacing=rating_spacing, skip_types=rating_skip_types)
    else
        fill(true, n_trials)
    end

    particles = [init_particle(n_trials, n_types) for _ in 1:n_particles]
    weights = fill(1.0 / n_particles, n_particles)
    log_weight_buffer = zeros(Float64, n_particles)

    cs_plus_mean = zeros(Float64, n_trials)
    cs_plus_sd = zeros(Float64, n_trials)
    cs_minus_mean = zeros(Float64, n_trials)
    cs_minus_sd = zeros(Float64, n_trials)
    trial_mean = zeros(Float64, n_trials)
    trial_sd = zeros(Float64, n_trials)

    cs_plus_stim_types = Int[1, 4]
    cs_plus_us_types = Int[4]
    cs_minus_stim_types = Int[2]
    cs_minus_us_types = Int[]

    for t in 1:n_trials
        cs_plus_preds = [particle_rating_prediction(p, cs_plus_stim_types, cs_plus_us_types,
                                                    t, alpha, a_obs, decay, omega, epsilon,
                                                    n_types, trial_types) for p in particles]
        cs_minus_preds = [particle_rating_prediction(p, cs_minus_stim_types, cs_minus_us_types,
                                                     t, alpha, a_obs, decay, omega, epsilon,
                                                     n_types, trial_types) for p in particles]

        mean_plus, sd_plus = weighted_stats(cs_plus_preds, weights)
        mean_minus, sd_minus = weighted_stats(cs_minus_preds, weights)
        cs_plus_mean[t] = mean_plus
        cs_plus_sd[t] = sd_plus
        cs_minus_mean[t] = mean_minus
        cs_minus_sd[t] = sd_minus

        trial_pred_values = if trial_types[t] == 1 || trial_types[t] == 4
            cs_plus_preds
        elseif trial_types[t] == 2
            cs_minus_preds
        else
            fill(0.5, length(cs_plus_preds))
        end
        tm, tsd = weighted_stats(trial_pred_values, weights)
        trial_mean[t] = tm
        trial_sd[t] = tsd

        for particle in particles
            propagate_particle!(particle, t, trial_types[t], alpha, a_obs, decay,
                                omega, epsilon, n_types, trial_types, rng)
        end

        for (idx, particle) in enumerate(particles)
            log_weight_buffer[idx] = particle.log_weight
        end
        log_mean = logsumexp(log_weight_buffer) - log(n_particles)
        for particle in particles
            particle.log_weight -= log_mean
        end

        weights = normalize_weights(particles)
        ess = 1.0 / sum(weights .^ 2)
        if ess < resample_threshold * n_particles
            particles, weights = resample_particles(particles, weights, rng)
        end
    end

    weights = normalize_weights(particles)

    rating_indices = findall(local_rating_mask)
    rating_trial_mean = trial_mean[rating_indices]
    rating_trial_sd = trial_sd[rating_indices]

    if compute_trajectories
        particle_assignments = zeros(Int, n_particles, n_trials)
        for p in 1:n_particles
            particle_assignments[p, :] = particles[p].causes
        end
    else
        particle_assignments = zeros(Int, 0, 0)
    end

    if compute_trajectories
        particle_highest = zeros(Int, n_particles, n_trials)
        for p in 1:n_particles
            particle_highest[p, :] = particles[p].highest_us_cause
        end
    else
        particle_highest = zeros(Int, 0, 0)
    end

    return Dict(
        "cs_plus_mean" => cs_plus_mean,
        "cs_plus_sd" => cs_plus_sd,
        "cs_minus_mean" => cs_minus_mean,
        "cs_minus_sd" => cs_minus_sd,
        "trial_mean" => trial_mean,
        "trial_sd" => trial_sd,
        "rating_mask" => local_rating_mask,
        "rating_indices" => rating_indices,
        "rating_trial_mean" => rating_trial_mean,
        "rating_trial_sd" => rating_trial_sd,
        "weights" => weights,
        "assignments" => particle_assignments,
        "highest_prob_cause" => particle_highest,
        "n_particles" => n_particles
    )
end


function resolve_model_parameters(parameters::Dict;
                                  randomize_missing::Bool=false,
                                  rng::AbstractRNG=Random.GLOBAL_RNG)
    
    params = Dict{Symbol, Any}(parameters)

    # Set defaults and issue warnings if necessary
    if !haskey(params, :α)
        @warn "Parameter α not provided. Defaulting to α = 1.0"
        params[:α] = 1.0
    end
    if !haskey(params, :a_obs)
        @warn "Parameter a_obs not provided. Defaulting to a_obs = 1.0."
        params[:a_obs] = 1.0
    end

    get_value(key::Symbol, default_val::Float64) = begin
        val = get(params, key, default_val)
        if val isa Distribution && !(val isa Dirac)
            return randomize_missing ? rand(rng, val) : default_val
        elseif val isa Dirac
            return val.value
        else
            return Float64(val)
        end
    end

    alpha = get_value(:α, 1.0)
    a_obs = get_value(:a_obs, 1.0)
    decay = get_value(:γ, 1.0)
    omega = get_value(:ω, 0.0)
    epsilon = get_value(:ε, 1.0)

    if haskey(params, :C) && haskey(params, :ω) && epsilon > 1.0
        @warn "Salience cannot be greater than 1 when selective maintenance is present. Setting ε = 1.0."
        epsilon = 1.0
    end

    return alpha, a_obs, decay, omega, epsilon
end


