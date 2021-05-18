_add_sum(x, y) = Base.add_sum(x, y)
_add_sum(x, ::Missing) = x
_add_sum(::Missing, x) = x
_add_sum(::Missing, ::Missing) = missing
_min_fun(x, y) = min(x, y)
_min_fun(x, ::Missing) = x
_min_fun(::Missing, y) = y
_min_fun(::Missing, ::Missing) = missing
_max_fun(x, y) = max(x, y)
_max_fun(x, ::Missing) = x
_max_fun(::Missing, y) = y
_max_fun(::Missing, ::Missing) = missing


function row_sum(f, df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    T = mapreduce(eltype, promote_type, eachcol(df)[colsidx])
    _op_for_sum!(x, y) = x .= _add_sum.(x, f.(y))
    # TODO the type of zeros after applying f???
    mapreduce(identity, _op_for_sum!, eachcol(df)[colsidx], init = zeros(T, nrow(df)))
end
row_sum(df::AbstractDataFrame, cols = :) = row_sum(identity, df, cols)

function row_count(df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    _op_for_count!(x, y) = x .+= .!ismissing.(y)
    mapreduce(identity, _op_for_count!, eachcol(df)[colsidx], init = zeros(Int32, nrow(df)))
end

# TODO not safe if the first column is Vector{Missing}
function row_minimum(f, df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    _op_for_min!(x, y) = x .= _min_fun.(x, f.(y))
    # TODO the type of zeros after applying f???
    mapreduce(identity, _op_for_min!, eachcol(df)[colsidx], init = f.(df[:, colsidx[1]]))
end
row_minimum(df::AbstractDataFrame, cols = :) = row_minimum(identity, df, cols)

# TODO not safe if the first column is Vector{Missing}
function row_maximum(f, df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    _op_for_max!(x, y) = x .= _max_fun.(x, f.(y))
    # TODO the type of zeros after applying f???
    mapreduce(identity, _op_for_max!, eachcol(df)[colsidx], init = f.(df[:, colsidx[1]]))
end
row_maximum(df::AbstractDataFrame, cols = :) = row_maximum(identity, df, cols)

function row_mean(f, df::AbstractDataFrame, cols = :)
    colsidx = DataFrames.index(df)[cols]
    T = mapreduce(eltype, promote_type, eachcol(df)[colsidx])
    _op_for_mean!(x, y) = (x[1] .= _add_sum.(x[1], f.(y)), x[2] .+= .!ismissing.(f.(y)))
    # TODO the type of zeros after applying f???
    rr = mapreduce(identity, _op_for_mean!, eachcol(df)[colsidx], init = (zeros(T, nrow(df)), zeros(Int32, nrow(df))))
    rr[1] ./ rr[2]
end
row_mean(df::AbstractDataFrame, cols = :) = row_mean(identity, df, cols)

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
function row_var(f, df::AbstractDataFrame, cols = :; dof = true)
    colsidx = DataFrames.index(df)[cols]
    T = mapreduce(eltype, promote_type, eachcol(df)[colsidx])
    _sq_(x) = x^2
    ss = row_sum(_sq_ ∘ f, df, cols)
    sval = row_sum(f, df, cols)
    n = row_count(df, cols)

    _row_wise_var(ss, sval, n, dof, T)
end
row_var(df::AbstractDataFrame, cols = :; dof = true) = row_var(identity, df, cols, dof = dof)


# function row_var(f, df::AbstractDataFrame, cols = :; dof = true)
#     colsidx = DataFrames.index(df)[cols]
#     T = mapreduce(eltype, promote_type, eachcol(df)[colsidx])
#     _op_for_var!(x, y) = (x[1] .= _add_sum.(x[1], f.(y) .* f.(y)), x[2] .= _add_sum.(x[2], f.(y)), x[3] .+= .!ismissing.(f.(y)))
#     # TODO the type of zeros after applying f???
#     rr = mapreduce(identity, _op_for_var!, eachcol(df)[colsidx], init = (zeros(T, nrow(df)), zeros(T, nrow(df)), zeros(Int32, nrow(df))))
#     _row_wise_var(rr[1], rr[2], rr[3], dof, T)
    
# end

function row_std(f, df::AbstractDataFrame, cols = :; dof = true)
    sqrt.(row_var(f, df, cols, dof = dof))
end
row_std(df::AbstractDataFrame, cols = :; dof = true) = row_std(identity, df, cols, dof = dof)

function row_stdze!(df::AbstractDataFrame , cols = :)
    meandata = row_mean(df, cols)
    stddata = row_std(df, cols)
    _stdze_fun(x) = (x .- meandata) ./ stddata 
    colsidx = DataFrames.index(df)[cols]

    for i in 1:length(colsidx)
        df[!, colsidx[i]] = _stdze_fun(df[!, colsidx[i]])
    end
end

