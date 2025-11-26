module ScreamTask

using Random
using Statistics
using StatsBase
using Distributions
using LogExpFunctions
using PRIMA
using BlackBoxOptim
using NOMAD
using ProgressLogging: @withprogress, @logprogress
using Base.Threads: Atomic, atomic_add!

include("particle_utils.jl")
include("simulate.jl")
include("optimize.jl")
include("plotting.jl")

export simulate, optimize!, plot_scream, intermittent_rating_mask

end # module ScreamTask
