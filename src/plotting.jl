using Plots
using Plots.PlotMeasures
using ColorSchemes
using LinearAlgebra: norm

function choose_distinct_palette(color_scheme::Symbol, n::Int=2; rng::AbstractRNG=Random.GLOBAL_RNG)
    cs = getfield(ColorSchemes, color_scheme)
    cols = cs.colors
    m = length(cols)
    m == 0 && error("Color scheme $color_scheme has no colours")
    if m <= n
        return cols[1:min(m, n)]
    end

    chosen = Vector{typeof(cols[1])}()
    push!(chosen, cols[rand(rng, 1:m)])
    while length(chosen) < n
        best_idx = 1
        best_dist = -Inf
        for (idx, col) in enumerate(cols)
            dist = minimum(norm.(Ref(col) .- chosen))
            if dist > best_dist
                best_dist = dist
                best_idx = idx
            end
        end
        push!(chosen, cols[best_idx])
    end
    return chosen
end

function _phase_spans()
    return [(:acquisition, PHASE_BOUNDARIES.acquisition),
            (:break1, PHASE_BOUNDARIES.break1),
            (:extinction, PHASE_BOUNDARIES.extinction),
            (:break2, PHASE_BOUNDARIES.break2),
            (:spontaneous_recovery, PHASE_BOUNDARIES.spontaneous_recovery),
            (:relearning, PHASE_BOUNDARIES.relearning)]
end

_phase_label_text(sym::Symbol) = replace(String(sym), "_" => "\n")

function _add_phase_guides!(plt; add_labels::Bool, label_height::Float64=1.05)
    for (phase, (start, stop)) in _phase_spans()
        span_start = start - 0.5
        span_stop = stop + 0.5
        if occursin("break", String(phase))
            vspan!(plt, [span_start, span_stop]; color=:gray, alpha=0.05, label="")
        end
        vline!(plt, [span_start]; line=:dash, color=:black, alpha=0.2, label="")
        if add_labels
            mid = start + (stop - start) / 2
            annotate!(plt, mid, label_height, Plots.text(_phase_label_text(phase), 8, :center))
        end
    end
end

function _modal_cause_per_trial(data::AbstractMatrix{<:Integer})
    n_trials = size(data, 2)
    n_trials == 0 && return Int[]
    modal_vals = zeros(Int, n_trials)
    for t in 1:n_trials
        freq = countmap(@view data[:, t])
        best_val = 0
        best_count = -1
        for (val, cnt) in freq
            if cnt > best_count || (cnt == best_count && val < best_val)
                best_val = val
                best_count = cnt
            end
        end
        modal_vals[t] = best_count < 0 ? 0 : best_val
    end
    return modal_vals
end

function _particle_line_plot(data::AbstractMatrix{<:Integer}, ylabel::String, panel_title::String;
                             color_scheme::Symbol,
                             opacity::Float64,
                             phase_labels::Bool)
    n_rows, n_trials = size(data)
    if n_trials == 0 || n_rows == 0
        return plot(title=panel_title, legend=false)
    end

    cs = getfield(ColorSchemes, color_scheme)
    color_cycle = Vector{eltype(cs.colors)}(undef, max(n_rows, 1))
    if n_rows <= 1
        color_cycle[1] = ColorSchemes.get(cs, 0.5)
    else
        for i in 1:n_rows
            frac = (i - 1) / max(n_rows - 1, 1)
            color_cycle[i] = ColorSchemes.get(cs, frac)
        end
    end

    trials = 1:n_trials
    cause_values = sort!(unique(vec(data)))
    cause_values = isempty(cause_values) ? [0] : cause_values

    plt = plot(xlabel="trial",
               ylabel=ylabel,
               yticks=(cause_values, string.(cause_values)),
               legend=false,
               title=panel_title)

    for idx in 1:n_rows
        plot!(plt, trials, data[idx, :];
              color=color_cycle[min(idx, length(color_cycle))],
              linewidth=1.2,
              linealpha=opacity,
              label="")
    end

    modal_vals = _modal_cause_per_trial(data)
    if !isempty(modal_vals)
        plot!(plt, trials, modal_vals;
              seriestype=:steppost,
              color=:black,
              linewidth=2,
              linealpha=0.4,
              linestyle=:dot,
              label="")
    end

    _add_phase_guides!(plt; add_labels=false)

    ylims!(plt, (minimum(cause_values) - 0.5, maximum(cause_values) + 0.5))
    return plt
end

function _parameter_value(params::AbstractDict, keys::Tuple{Vararg{Symbol}})
    for key in keys
        if haskey(params, key)
            return Float64(params[key])
        end
    end
    error("Missing parameter $(keys[1]) in simulation entry; rerun `simulate` with parameter storage enabled.")
end

function _ensure_particle_details(entry::AbstractDict;
                                   n_particles::Int,
                                   resample_threshold::Float64,
                                   seed::Union{Nothing, Int})
    existing = get(entry, :particle_filter, nothing)
    if existing !== nothing && size(existing["assignments"], 2) == length(entry[:trial_types])
        return existing
    end

    params = get(entry, :parameters, nothing)
    params === nothing && error("Full simulation data must include :parameters to reconstruct particle trajectories.")
    trial_types = entry[:trial_types]
    mask = get(entry, :rating_mask, nothing)
    participant = get(entry, :participant, 0)
    pf_seed = seed === nothing ? (participant isa Integer ? Int(participant) : 0) : seed

    alpha = _parameter_value(params, (:α, :alpha))
    a_obs = _parameter_value(params, (:a_obs,))
    decay = _parameter_value(params, (:γ, :gamma))
    omega = _parameter_value(params, (:ω, :omega))
    epsilon = _parameter_value(params, (:ε, :epsilon))

    pf = run_particle_filter(trial_types, alpha, a_obs, decay, omega, epsilon;
                             n_particles=n_particles,
                             resample_threshold=resample_threshold,
                             seed=pf_seed,
                             compute_trajectories=true,
                             intermittent=false,
                             rating_mask=mask)

    if entry isa Dict
        entry[:particle_filter] = pf
    end
    return pf
end

function _format_param_subtitle(param_dict)
    param_dict === nothing && return ""
    pairs = collect(param_dict)
    isempty(pairs) && return ""
    return join(["$(string(k))=$(round(Float64(v), digits=3))" for (k, v) in sort(pairs; by=x -> string(x[1]))], " | ")
end

function _single_task_plot(entry::AbstractDict;
                           color_scheme::Symbol,
                           plot_size::Tuple{Int, Int},
                           phase_labels::Bool,
                           pf_particles::Int,
                           pf_resample_threshold::Float64,
                           pf_seed::Union{Nothing, Int},
                           cause_opacity::Float64)
    pf_result = _ensure_particle_details(entry;
                                         n_particles=pf_particles,
                                         resample_threshold=pf_resample_threshold,
                                         seed=pf_seed)

    cs_plus_mean = pf_result["cs_plus_mean"]
    cs_plus_sd = pf_result["cs_plus_sd"]
    cs_minus_mean = pf_result["cs_minus_mean"]
    cs_minus_sd = pf_result["cs_minus_sd"]
    trial_types = entry[:trial_types]
    trials = 1:length(trial_types)
    participant = get(entry, :participant, "?")
    params = get(entry, :parameters, nothing)

    palette = choose_distinct_palette(color_scheme, 2)
    cs_plus_color, cs_minus_color = palette

    title = "participant $(participant)"
    subtitle = _format_param_subtitle(params)

    p_pred = plot(trials, cs_plus_mean;
                  ribbon=cs_plus_sd,
                  fillalpha=0.2,
                  label="p(US | CS+)",
                  linewidth=2,
                  color=cs_plus_color,
                  xlabel="trial",
                  ylabel="p(US)",
                  ylims=(-0.05, 1.15),
                  legend=(0.85, 0.8),
                  title=title,
                  margins=5mm)
    plot!(p_pred, trials, cs_minus_mean;
          ribbon=cs_minus_sd,
          fillalpha=0.2,
          linestyle=:dash,
          linewidth=2,
          label="p(US | CS-)",
          color=cs_minus_color)

    us_trials = findall(x -> x == 4, trial_types)
    if !isempty(us_trials)
        scatter!(p_pred, us_trials, cs_plus_mean[us_trials];
                 marker=:star5,
                 markersize=6,
                 color=:black,
                 label="US presented")
    end

    if !isempty(subtitle)
        annotate!(p_pred, mean(trials), 1.2, Plots.text(subtitle, 9, :center))
    end

    if phase_labels
        _add_phase_guides!(p_pred; add_labels=true, label_height=1.08)
    else
        _add_phase_guides!(p_pred; add_labels=false)
    end

    assignments = pf_result["assignments"]
    highest = pf_result["highest_prob_cause"]

    panels = Any[p_pred]
    if size(assignments, 2) > 0
        push!(panels, _particle_line_plot(assignments, "latent cause", "particle cause trajectories";
                                          color_scheme=color_scheme,
                                          opacity=cause_opacity,
                                          phase_labels=phase_labels))
    end
    if size(highest, 2) > 0
        push!(panels, _particle_line_plot(highest, "latent cause", "highest p(US) cause";
                                          color_scheme=color_scheme,
                                          opacity=cause_opacity,
                                          phase_labels=phase_labels))
    end

    if length(panels) == 1
        return plot(panels...; size=plot_size)
    else
        heights = fill(0.2, length(panels))
        heights[1] = 0.6
        remaining = length(panels) - 1
        if remaining > 0
            heights[2:end] .= (1 - heights[1]) / remaining
        end
        return plot(panels...; layout=grid(length(panels), 1, heights=heights), size=plot_size)
    end
end

function _aggregate_task_plot(data::Vector{<:AbstractDict};
                              color_scheme::Symbol,
                              plot_size::Tuple{Int, Int},
                              phase_labels::Bool)
    cs_plus_matrix = reduce(hcat, [entry[:cs_plus_mean] for entry in data])
    cs_minus_matrix = reduce(hcat, [entry[:cs_minus_mean] for entry in data])
    trials = 1:size(cs_plus_matrix, 1)

    mean_plus = mean(cs_plus_matrix; dims=2)[:]
    se_plus = std(cs_plus_matrix; dims=2, corrected=false)[:]
    se_plus ./= sqrt(size(cs_plus_matrix, 2))

    mean_minus = mean(cs_minus_matrix; dims=2)[:]
    se_minus = std(cs_minus_matrix; dims=2, corrected=false)[:]
    se_minus ./= sqrt(size(cs_minus_matrix, 2))

    palette = choose_distinct_palette(color_scheme, 2)
    cs_plus_color, cs_minus_color = palette

    p = plot(trials, mean_plus;
             ribbon=se_plus,
             fillalpha=0.2,
             linewidth=2,
             color=cs_plus_color,
             label="mean CS+",
             xlabel="trial",
             ylabel="p(US)",
             ylims=(-0.05, 1.05),
             title="mean predicted scream ratings across participants",
             size=plot_size,
             margins=8mm)
    plot!(p, trials, mean_minus;
          ribbon=se_minus,
          fillalpha=0.2,
          linewidth=2,
          linestyle=:dash,
          color=cs_minus_color,
          label="mean CS-")

    _add_phase_guides!(p; add_labels=phase_labels)

    return p
end

function _plot_task_data(data;
                         individual::Bool,
                         participant_no::Int,
                         color_scheme::Symbol,
                         plot_size::Tuple{Int, Int},
                         _rng::AbstractRNG,
                         phase_labels::Bool,
                         pf_particles::Int,
                         pf_resample_threshold::Float64,
                         cause_opacity::Float64,
                         pf_seed::Union{Nothing, Int})
    entries = data isa AbstractVector ? collect(data) : [data]
    isempty(entries) && error("No data provided for plotting.")
    required_keys = (:cs_plus_mean, :cs_minus_mean, :trial_types)
    for entry in entries
        all(key -> haskey(entry, key), required_keys) || error("Full simulation data must contain keys $(required_keys).")
    end

    if individual
        effective_idx = participant_no
        effective_idx = clamp(effective_idx, 1, length(entries))
        local_seed = pf_seed === nothing ? nothing : pf_seed + effective_idx - 1
        return _single_task_plot(entries[effective_idx];
                                 color_scheme=color_scheme,
                                 plot_size=plot_size,
                                 phase_labels=phase_labels,
                                 pf_particles=pf_particles,
                                 pf_resample_threshold=pf_resample_threshold,
                                 pf_seed=local_seed,
                                 cause_opacity=cause_opacity)
    else
        return _aggregate_task_plot(entries;
                                    color_scheme=color_scheme,
                                    plot_size=plot_size,
                                    phase_labels=phase_labels)
    end
end

function _resolve_true_parameters(results, provided)
    if provided !== nothing
        return provided
    end
    return [res.true_parameters for res in results]
end

function _plot_recovery_data(results;
                             true_parameters,
                             free_parameters,
                             plot_size::Tuple{Int, Int})
    isempty(results) && error("No recovery results supplied for plotting.")
    keys = free_parameters === nothing ? collect(keys(results[1].estimate)) : Symbol.(free_parameters)
    true_sets = _resolve_true_parameters(results, true_parameters)
    length(true_sets) == length(results) || error("true_parameters must match the number of recovery results.")

    panels = Vector{Any}(undef, length(keys))
    for (idx, key) in enumerate(keys)
        true_vals = Float64[]
        est_vals = Float64[]
        for (res, truth) in zip(results, true_sets)
            truth === nothing && continue
            if !haskey(truth, key)
                continue
            end
            push!(true_vals, truth[key])
            push!(est_vals, res.estimate[key])
        end
        isempty(true_vals) && error("True parameter values for $(key) are unavailable.")
        corr_val = length(true_vals) > 1 ? cor(true_vals, est_vals) : NaN
        lbl = isnan(corr_val) ? "r = NaN" : "r = $(round(corr_val, digits=3))"
        panels[idx] = scatter(true_vals, est_vals;
                              xlabel="true $(key)",
                              ylabel="estimated $(key)",
                              label=lbl,
                              legend=:bottomright,
                              markersize=5,
                              color=:steelblue,
                              title=string(key))
        min_val = minimum(vcat(true_vals, est_vals))
        max_val = maximum(vcat(true_vals, est_vals))
        plot!(panels[idx], [min_val, max_val], [min_val, max_val]; linestyle=:dash, color=:black, label="")
    end

    return plot(panels...; layout=(length(panels), 1), size=plot_size)
end

function plot_scream(data; type::String="task", individual::Bool=true,
              participant_no::Int=1, color_scheme::Symbol=:seaborn_pastel6,
              plot_size::Tuple{Int, Int}=(900, 600), _rng::AbstractRNG=Random.GLOBAL_RNG,
              true_parameters=nothing, free_parameters=nothing, phase_labels::Bool=true,
              pf_particles::Int=500, pf_resample_threshold::Float64=0.5,
              cause_opacity::Float64=0.1, pf_seed::Union{Nothing, Int}=nothing)
    if type == "task"
        return _plot_task_data(data; individual=individual,
                               participant_no=participant_no,
                               color_scheme=color_scheme,
                               plot_size=plot_size, _rng=_rng,
                               phase_labels=phase_labels,
                               pf_particles=pf_particles,
                               pf_resample_threshold=pf_resample_threshold,
                               cause_opacity=cause_opacity, pf_seed=pf_seed)
    elseif type == "recovery"
        results = data isa AbstractVector ? collect(data) : [data]
        return _plot_recovery_data(results; true_parameters=true_parameters,
                                   free_parameters=free_parameters, plot_size=plot_size)
    else
        error("Unknown plot type $(type). Use `task` or `recovery`.")
    end
end
