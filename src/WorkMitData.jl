module WorkMitData


using Reexport
using Dates
using Random
using PooledArrays
@reexport using DataFrames
using Statistics
@reexport using StatsBase

export
	lag,
	lead,
	dttodate,
	dttodate!,
	stdze,
	rescale,
	intck,
	maximum,
	minimum,
	sum,
	weightedsum,
	weightedmean,
	mean,
	var,
	std,
	median,
	quantile,
	skipnan,
	k_largest,
	k_smallest,
	# from extra_df.jl
	std_names!,
	row_sum,
	row_mean,
	# from reshape.jl
	t_function


include("extra_fun.jl")
include("extra_df.jl")
include("reshape.jl")


end
