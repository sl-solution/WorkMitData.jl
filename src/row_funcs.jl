_add_sum(x, y) = Base.add_sum(x, y)
_add_sum(x, ::Missing) = x
_add_sum(::Missing, x) = x
_add_sum(::Missing, ::Missing) = missing
_mul_prod(x, y) = Base.mul_prod(x, y)
_mul_prod(x, ::Missing) = x
_mul_prod(::Missing, x) = x
_mul_prod(::Missing, ::Missing) = missing
_min_fun(x, y) = min(x, y)
_min_fun(x, ::Missing) = x
_min_fun(::Missing, y) = y
_min_fun(::Missing, ::Missing) = missing
_max_fun(x, y) = max(x, y)
_max_fun(x, ::Missing) = x
_max_fun(::Missing, y) = y
_max_fun(::Missing, ::Missing) = missing

_bool(f) = x->f(x)::Bool

struct _Prehashed
    hash::UInt64
end
Base.hash(x::_Prehashed) = x.hash

"""
    row_sum([f = identity,] df::AbstractDataFrame[, cols])
    computes the sum of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_sum(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zero(CT)))
    if CT >: Missing
        T = Union{Missing, T}
    end
    _op_for_sum!(x, y) = x .= _add_sum.(x, f.(y))
    init0 = fill!(Vector{T}(undef, nrow(df)), T >: Missing ? missing : zero(T))
    mapreduce(identity, _op_for_sum!, view(getfield(df, :columns),colsidx), init = init0)
end
row_sum(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_sum(identity, df, cols)

"""
    row_prod([f = identity,] df::AbstractDataFrame[, cols])
    computes the product of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_prod(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zero(CT)))
    if CT >: Missing
        T = Union{Missing, T}
    end
    _op_for_prod!(x, y) = x .= _mul_prod.(x, f.(y))
    init0 = fill!(Vector{T}(undef, nrow(df)), T >: Missing ? missing : one(T))
    mapreduce(identity, _op_for_prod!, view(getfield(df, :columns),colsidx), init = init0)
end
row_prod(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_prod(identity, df, cols)

"""
    row_count(f, df::AbstractDataFrame[, cols])
    counts the number of non-missing values in each row of df[!, cols] for which the function `f` returns `true`.
"""
function row_count(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    _op_for_count!(x, y) = x .+= (_bool(f).(y))
    mapreduce(identity, _op_for_count!, view(getfield(df, :columns),colsidx), init = zeros(Int32, nrow(df)))
end
row_count(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_count(x->true, df, cols)

# """
#     row_anymissing(df::AbstractDataFrame[, cols])
#     returns `true` or `false` wheather the row contains `missing` or no `missing`, respectively.
# """
# function row_anymissing(df::AbstractDataFrame, cols = :)
#     colsidx = DataFrames.index(df)[cols]
#     # sel_colsidx = findall(x-> x >: Missing, eltype.(eachcol(df)[colsidx]))
#     _op_bool_add(x::Bool,y::Bool) = x || y ? true : false
#     op_for_anymissing!(x,y) = x .= _op_bool_add.(x, _bool(ismissing).(y))
#     # mapreduce(identity, op_for_anymissing!, eachcol(df)[colsidx[sel_colsidx]], init = zeros(Bool, nrow(df)))
#     mapreduce(identity, op_for_anymissing!, view(getfield(df, :columns),colsidx), init = zeros(Bool, nrow(df)))
# end


function row_any(f, df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    _op_bool_add(x::Bool,y::Bool) = x || y ? true : false
    op_for_any!(x,y) = x .= _op_bool_add.(x, _bool(f).(y))
    # mapreduce(identity, op_for_anymissing!, eachcol(df)[colsidx[sel_colsidx]], init = zeros(Bool, nrow(df)))
    mapreduce(identity, op_for_any!, view(getfield(df, :columns),colsidx), init = zeros(Bool, nrow(df)))
end
row_any(df::AbstractDataFrame, cols = :) = row_any(x->true, df, cols)

function row_all(f, df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    _op_bool_mult(x::Bool,y::Bool) = x && y ? true : false
    op_for_all!(x,y) = x .= _op_bool_mult.(x, _bool(f).(y))
    # mapreduce(identity, op_for_anymissing!, eachcol(df)[colsidx[sel_colsidx]], init = zeros(Bool, nrow(df)))
    mapreduce(identity, op_for_all!, view(getfield(df, :columns),colsidx), init = ones(Bool, nrow(df)))
end
row_all(df::AbstractDataFrame, cols = :) = row_all(x->true, df, cols)

"""
    row_mean([f = identity,] df::AbstractDataFrame[, cols])
    computes the mean of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_mean(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    row_sum(f, df, cols) ./ row_count(x -> !ismissing(x), df, cols)
end
row_mean(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_mean(identity, df, cols)

# TODO not safe if the first column is Vector{Missing}
"""
    row_minimum([f = identity,] df::AbstractDataFrame[, cols])
    finds the minimum of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_minimum(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zeros(CT)[1]))
    if CT >: Missing
        T = Union{Missing, T}
    end
    _op_for_min!(x, y) = x .= _min_fun.(x, f.(y))
    init0 = fill!(Vector{T}(undef, nrow(df)), T >: Missing ? missing : typemax(T))
    mapreduce(identity, _op_for_min!, view(getfield(df, :columns),colsidx), init = init0)
end
row_minimum(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_minimum(identity, df, cols)

# TODO not safe if the first column is Vector{Missing}
"""
    row_maximum([f = identity,] df::AbstractDataFrame[, cols])
    finds the maximum of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_maximum(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zeros(CT)[1]))
    if CT >: Missing
        T = Union{Missing, T}
    end
    _op_for_max!(x, y) = x .= _max_fun.(x, f.(y))
    # TODO the type of zeros after applying f???
    init0 = fill!(Vector{T}(undef, nrow(df)), T >: Missing ? missing : typemin(T))
    mapreduce(identity, _op_for_max!, view(getfield(df, :columns),colsidx), init = init0)
end
row_maximum(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_maximum(identity, df, cols)

# TODO better function for the first component of operator
function _row_wise_var(ss, sval, n, dof, T)
    res = Vector{T}(undef, length(ss))
    for i in 1:length(ss)
        if n[i] == 0
            res[i] = missing
        elseif n[i] == 1
            res[i] = zero(T)
        else
            res[i] = ss[i]/n[i] - (sval[i]/n[i])*(sval[i]/n[i])
            if dof
                res[i] = (n[i] * res[i])/(n[i]-1)
            end
        end
    end
    res
end

# TODO needs type stability
"""
    row_var([f = identity,] df::AbstractDataFrame[, cols]; dof = true)
    computes the variance of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_var(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); dof = true)
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zero(CT)))
    if CT >: Missing
        T = Union{Missing, T}
    end
    _sq_(x) = x^2
    ss = row_sum(_sq_ ∘ f, df, cols)
    sval = row_sum(f, df, cols)
    n = row_count(x -> !ismissing(x), df, cols)
    res = ss ./ n .- (sval ./ n) .^ 2
    if dof
        res .= (n .* res) ./ (n .- 1)
        res .= ifelse.(n .== 1, zero(T), res)
    end
    res
    # _row_wise_var(ss, sval, n, dof, T)
end
row_var(df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); dof = true) = row_var(identity, df, cols, dof = dof)


# function row_var(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); dof = true)
#     colsidx = DataFrames.index(df)[cols]
#     T = mapreduce(eltype, promote_type, eachcol(df)[colsidx])
#     _op_for_var!(x, y) = (x[1] .= _add_sum.(x[1], f.(y) .* f.(y)), x[2] .= _add_sum.(x[2], f.(y)), x[3] .+= .!ismissing.(f.(y)))
#     # TODO the type of zeros after applying f???
#     rr = mapreduce(identity, _op_for_var!, eachcol(df)[colsidx], init = (zeros(T, nrow(df)), zeros(T, nrow(df)), zeros(Int32, nrow(df))))
#     _row_wise_var(rr[1], rr[2], rr[3], dof, T)

# end
"""
    row_std([f = identity,] df::AbstractDataFrame[, cols]; dof = true)
    computes the standard deviation of non-missing values in each row of df[!, cols] after applying `f` on each value.
"""
function row_std(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); dof = true)
    sqrt.(row_var(f, df, cols, dof = dof))
end
row_std(df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); dof = true) = row_std(identity, df, cols, dof = dof)

function row_cumsum!(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zeros(CT)[1]))
    for i in 2:length(colsidx)
        if eltype(df[!, colsidx[i]]) >: Missing
            df[!, colsidx[i]] = convert(Vector{Union{Missing, T}}, df[!, colsidx[i]])
        else
            df[!, colsidx[i]] = convert(Vector{T}, df[!, colsidx[i]])
        end
    end

    _op_for_cumsum!(x, y) = y .= _add_sum.(x, f.(y))

    CT = eltype(df[!, colsidx[1]])
    T = typeof(f(zeros(CT)[1]))
    if CT >: Missing
        T = Union{Missing, T}
    end
    init0 = fill!(Vector{T}(undef, nrow(df)), T >: Missing ? missing : zero(T))
    mapreduce(identity, _op_for_cumsum!, view(getfield(df, :columns),colsidx), init = init0)
    nothing
end
row_cumsum!(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_cumsum!(identity, df, cols)

function row_cumsum(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    dfcopy = copy(df)
    row_cumsum!(f, dfcopy, cols)
    dfcopy
end


function row_cumprod!(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    colsidx = DataFrames.index(df)[cols]
    CT = mapreduce(eltype, promote_type, view(getfield(df, :columns),colsidx))
    T = typeof(f(zeros(CT)[1]))
    for i in 2:length(colsidx)
        if eltype(df[!, colsidx[i]]) >: Missing
            df[!, colsidx[i]] = convert(Vector{Union{Missing, T}}, df[!, colsidx[i]])
        else
            df[!, colsidx[i]] = convert(Vector{T}, df[!, colsidx[i]])
        end
    end
    _op_for_cumprod!(x, y) = y .= _mul_prod.(x, f.(y))
    CT = eltype(df[!, colsidx[1]])
    T = typeof(f(zeros(CT)[1]))
    if CT >: Missing
        T = Union{Missing, T}
    end
    init0 = fill!(Vector{T}(undef, nrow(df)), T >: Missing ? missing : one(T))
    mapreduce(identity, _op_for_cumprod!, view(getfield(df, :columns),colsidx), init = init0)
    nothing
end
row_cumprod!(df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_cumprod!(identity, df, cols)

function row_cumprod(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}))
    dfcopy = copy(df)
    row_cumprod!(f, dfcopy, cols)
    dfcopy
end


"""
    row_stdze!(df::AbstractDataFrame[, cols])
    standardised the values within each row of df[!, cols], and replaces the old values.
"""
function row_stdze!(df::AbstractDataFrame , cols = names(df, Union{Missing, Number}))
    meandata = row_mean(df, cols)
    stddata = row_std(df, cols)
    _stdze_fun(x) = ifelse.(isequal.(stddata, 0), missing, (x .- meandata) ./ stddata)
    colsidx = DataFrames.index(df)[cols]

    for i in 1:length(colsidx)
        df[!, colsidx[i]] = _stdze_fun(df[!, colsidx[i]])
    end
end

"""
    row_stdze(df::AbstractDataFrame[, cols])
    standardised the values within each row of df[!, cols].
"""
function row_stdze(df::AbstractDataFrame , cols = names(df, Union{Missing, Number}))
    dfcopy = copy(df)
    row_stdze!(dfcopy, cols)
    dfcopy
end

"""
    row_sort!(df::AbstractDataFrame[, cols]; kwargs...)
    replace `cols` in each row with their sorted values.
"""
function row_sort!(df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); kwargs...)
    colsidx = DataFrames.index(df)[cols]
    T = mapreduce(eltype, promote_type, eachcol(df)[colsidx])
    m = Matrix{T}(df[!, colsidx])
    sort!(m; dims = 2, kwargs...)
    for i in 1:length(colsidx)
        getfield(df, :columns)[colsidx[i]] = m[:, i]
    end
end

"""
    row_sort!(df::AbstractDataFrame[, cols]; kwargs...)
    sort `cols` in each row.
"""
function row_sort(df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); kwargs...)
    dfcopy = copy(df)
    row_sort!(dfcopy, cols; kwargs...)
    dfcopy
end

# TODO is it possible to have a faster row_count_unique??
function _fill_prehashed!(prehashed, y, f, n, j)
    @views copy!(prehashed[:, j] , _Prehashed.(hash.(f.(y))))
end

function _fill_dict_and_add!(init0, dict, prehashed, n, p)
    for i in 1:n
        for j in 1:p
            if !haskey(dict, prehashed[i, j])
                get!(dict, prehashed[i, j], nothing)
                init0[i] += 1
            end
        end
        empty!(dict)
    end
end

"""
    row_nunique([f = identity,] df::AbstractDataFrame[, cols]; count_missing = true)
    count the number of unique values in each row of df[!, cols] after applying `f` on each value. If `count_missing = false`, `missing` are not counted.
"""
function row_nunique(f, df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); count_missing = true)
    colsidx = DataFrames.index(df)[cols]
    prehashed = Matrix{_Prehashed}(undef, nrow(df), length(colsidx))
    allcols = view(getfield(df, :columns),colsidx)

    for j in 1:size(prehashed,2)
        _fill_prehashed!(prehashed, allcols[j], f, nrow(df), j)
    end

    init0 = zeros(Int32, nrow(df))
    dict = Dict{_Prehashed, Nothing}()
    _fill_dict_and_add!(init0, dict, prehashed, nrow(df), length(colsidx))
    if count_missing
        return init0
    else
        return init0 .- row_any(ismissing, df, cols)
    end
end
row_nunique(df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); count_missing = true) = row_nunique(identity, df, cols; count_missing = count_missing)

struct _DUMMY_STRUCT
end

# anymissing(::_DUMMY_STRUCT) = false
nunique(::_DUMMY_STRUCT) =  false
stdze!(::_DUMMY_STRUCT) = false
stdze(::_DUMMY_STRUCT) = false


byrow(::typeof(sum), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_sum(by, df, cols)

byrow(::typeof(prod), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_prod(by, df, cols)

byrow(::typeof(count), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = x->true) = row_count(by, df, cols)

# byrow(::typeof(anymissing), df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_anymissing(df, cols)

byrow(::typeof(any), df::AbstractDataFrame, cols = :; by = x->true) = row_any(by, df, cols)

byrow(::typeof(all), df::AbstractDataFrame, cols = :; by = x->true) = row_all(by, df, cols)

byrow(::typeof(mean), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_mean(by, df, cols)

byrow(::typeof(maximum), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_maximum(by, df, cols)

byrow(::typeof(minimum), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_minimum(by, df, cols)

byrow(::typeof(var), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity, dof = true) = row_var(by, df, cols; dof = dof)

byrow(::typeof(std), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity, dof = true) = row_std(by, df, cols; dof = dof)

byrow(::typeof(nunique), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity, count_missing = true) = row_nunique(by, df, cols; count_missing = count_missing)

byrow(::typeof(cumsum), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_cumsum(by, df, cols)

byrow(::typeof(cumprod!), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_cumprod!(by, df, cols)

byrow(::typeof(cumprod), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_cumprod(by, df, cols)

byrow(::typeof(cumsum!), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); by = identity) = row_cumsum!(by, df, cols)

byrow(::typeof(sort), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); kwargs...) = row_sort(df, cols; kwargs...)

byrow(::typeof(sort!), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); kwargs...) = row_sort!(df, cols; kwargs...)

byrow(::typeof(stdze), df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_stdze(df, cols)

byrow(::typeof(stdze!), df::AbstractDataFrame, cols = names(df, Union{Missing, Number})) = row_stdze!(df, cols)

byrow(::typeof(mapreduce), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); op = .+, kwargs...) = mapreduce(identity, op, eachcol(df[!, cols]); kwargs...)

byrow(::typeof(reduce), df::AbstractDataFrame, cols = names(df, Union{Missing, Number}); op = .+, kwargs...) = reduce(op, eachcol(df[!, cols]); kwargs...)

byrow(f::Function, df::AbstractDataFrame, cols) = f.(eachrow(df[!, cols]))
