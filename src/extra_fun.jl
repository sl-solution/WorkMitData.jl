"""
skipnan(x, init=zero(x)) replace NaN with a neutral value for some operators. E.g init should be set to zero(x) for + and it should be set to one(x) for * operator. This is useful when working with IndexedTables.jl
"""
skipnan(x, init = zero(x)) = isnan(x) ? init : x

# TODO take care of performance
"""
intck(x,y) computes the difference between two date objects and return number of day. Any of x and y can be missing value.
"""
intck(::Missing,::Missing) = missing
intck(::Missing,y) = missing
intck(x,::Missing) = missing
intck(x::Date,y::Date) = (x-y).value

"""
rescale(x,minx,maxx,minval,maxval) rescales x to run from minval and maxval, given x originaly runs from minx to maxx.
"""
function rescale(x,minx,maxx,minval,maxval)
    -(-maxx*minval+minx*maxval)/(maxx-minx)+(-minval+maxval)*x/(maxx-minx)
end
rescale(::Missing,minx,maxx,minval,maxval) = missing
rescale(x::Vector,minx,maxx,minval,maxval) = rescale.(x,minx,maxx,minval,maxval)
rescale(x,minx,maxx) = rescale(x,minx,maxx,0.0,1.0)

"""
stdze(x) standardizes an array. It return missing for missing data points.
"""
function stdze(x)
    all(ismissing,x) && return x
    meandata = mean(x)
    vardata = var(x)
    (x .- meandata) ./ sqrt(vardata)
end



"""
lag(x,k) Creates a lag-k of the provided array x. The output will be an array the
same size as x (the input array), and the its type will be Union{Missing, T} where T is the type of input.
"""
function lag(x,k)
    res=zeros(Union{eltype(x),Missing},length(x))
    @simd for i in 1:k
        @inbounds res[i] = missing
    end
    @simd for i in k+1:length(x)
        @inbounds res[i] = x[i-k]
    end
    res
end

lag(x)=lag(x,1)

"""
lead(x,k) Creates a lead-k of the provided array x. The output will be an array the
same size as x (the input array), and the its type will be Union{Missing, T} where T is the type of input.
"""
function lead(x,k)
    res=zeros(Union{eltype(x),Missing},length(x))
    @simd for i in 1:length(x)-k
        @inbounds res[i] = x[i+k]
    end
    @simd for i in length(x)-k+1:length(x)
        @inbounds res[i] = missing
    end
    res
end
lead(x)=lead(x,1)

"""
dttodate(x) converts SAS or STATA dates (which is the number of days after 01-01-1960) to a Julia Date object.
dttodate(DataFrame,cols) converts the given columns to Date object.
"""
dttodate(::Missing) = missing
dttodate(x) = Date(1960,1,1)+Day(x)
dttodate(x::Date) = x
function dttodate!(df::DataFrame,cols)
    for i in cols
        df[!,i] = dttodate.(df[!,i])
    end
end

# modifying some Base functions to suit for working with data with missing values

# in mapreduce when f change the type of data the performance is affected, TODO is there any way to avoid it???
Base.maximum(f, x::AbstractArray{Missing,1})=missing
function Base.maximum(f, x::AbstractArray{Union{T,Missing},1}) where {T <: Number}
    all(ismissing, x) && return missing
    _dmiss(x)::T = ismissing(f(x)) ? typemin(T) : f(x)
    mapreduce(_dmiss,max, x)
end
Base.maximum(x::AbstractArray{Union{T,Missing},1}) where {T <: Number} = maximum(identity, x)

Base.maximum(x::AbstractArray{Union{Date,Missing},1})=maximum(skipmissing(x))

Base.minimum(f, ::AbstractArray{Missing,1})=missing
function Base.minimum(f, x::AbstractArray{Union{T,Missing},1}) where {T <: Number}
    all(ismissing, x) && return missing
    @inline _dmiss(x)::T = ismissing(f(x)) ? typemax(T) : f(x)
    mapreduce(_dmiss,min, x)
end
Base.minimum(x::AbstractArray{Union{T,Missing},1}) where {T <: Number} = minimum(identity, x)
Base.minimum(x::AbstractArray{Union{Date,Missing},1})=minimum(skipmissing(x))


Base.sum(f, ::AbstractArray{Missing,1}) = missing
function Base.sum(f, x::AbstractArray{Union{T,Missing},1}) where {T <: Number}
    all(ismissing, x) && return missing
    _dmiss(y) = ifelse(ismissing(f(y)),  zero(T), f(y))
    mapreduce(_dmiss, +, x)
end
Base.sum(x::AbstractArray{Union{T,Missing},1}) where {T <: Number} = sum(identity, x)

weightedsum(f, ::AbstractArray{Missing,1}, ::AbstractArray{Missing,1}) = missing
function weightedsum(f, x::AbstractArray{Union{T,Missing},1}, w) where {T <: Number}
    all(ismissing, x) && return missing
    _dmiss(y)::T = ismissing(y[1])||ismissing(y[2]) ? zero(T) : (f(y[1])*y[2])::T
    mapreduce(_dmiss, +, zip(x,w))
end
weightedsum(x::AbstractArray{Union{T,Missing},1}, w) where {T <: Number}  = weightedsum(identity, x, w)

function _countnonmissing(x)
    res = 0
    @simd for i in x
        res += !ismissing(i)
    end
    res
end

Statistics.mean(f, ::AbstractArray{Missing,1}) = missing

function Statistics.mean(f, x::AbstractArray{Union{T,Missing},1}) where {T <: Number}
    _op(y1,y2) = (y1[1]+y2[1],y1[2]+y2[2])::Tuple{T,Int}
    _dmiss(y) = (ismissing(f(y)) ? zero(T) : f(y), !ismissing(f(y)))::Tuple{T,Bool}
    sval, n = mapreduce(_dmiss, _op, x)::Tuple{T, Int}
    n == 0 ? missing : sval/n
end

Statistics.mean(x::AbstractArray{Union{T,Missing},1}) where {T <: Number} = mean(identity, x)

weightedmean(f, ::AbstractArray{Missing,1}, ::AbstractArray{Missing,1}) = missing
function weightedmean(f, x::AbstractArray{Union{T,Missing},1}, w::AbstractArray{S,1}) where {T <: Number} where {S <: Number}
    all(ismissing, x) && return missing
    _dmiss(y)::T = ismissing(y[1])||ismissing(y[2]) ? zero(T) : (f(y[1])*y[2])::T
    _dmiss2(y)::S = ismissing(y[1])||ismissing(y[2]) ? zero(S) : y[2]
    _op(y1,y2)::Tuple{T,S} = y1 .+ y2
    _f(y)::Tuple{T,S} = (_dmiss(y), _dmiss2(y))
    sval, n = mapreduce(_f, _op, zip(x,w))::Tuple{T,S}
    n == 0 ? missing : sval / n
end
weightedmean(x::AbstractArray{Union{T,Missing},1}, w::AbstractArray{S,1}) where {T <: Number} where {S <: Number} = weightedmean(identity, x, w)


Statistics.var(f, x::AbstractArray{Missing,1}, df=true)=missing
function Statistics.var(f, x::AbstractArray{Union{T,Missing},1}, df=true) where {T <: Number}
    _opvar(y1,y2) = (y1[1]+y2[1], y1[2]+y2[2], y1[3]+y2[3])::Tuple{T,T,Int}
    _dmiss(y) = ismissing(f(y)) ? zero(T) : f(y)
    _varf(y) = (_dmiss(y) ^ 2, _dmiss(y) , !ismissing(f(y)))

    ss, sval, n = mapreduce(_varf, _opvar, x)::Tuple{T, T, Int}

    if n == 0
        return missing
    elseif n == 1
        return zero(T)
    else
        res = ss/n - (sval/n)*(sval/n)
        if df
            return (n * res)/(n-1)
        else
            return res
        end
    end
end

Statistics.var( x::AbstractArray{Union{T,Missing},1}, df=true) where {T <: Number} = var(identity, x, df)

Statistics.std(f, x::AbstractArray{Missing,1}, df=true)=missing
function Statistics.std(f , x::AbstractArray{Union{T,Missing},1}, df=true) where {T <: Number}
    sqrt(var(f, x,df))
end
Statistics.std(x::AbstractArray{Union{T,Missing},1}, df=true) where {T <: Number} = std(identity, x, df)


function Statistics.median(v::AbstractArray{Union{T,Missing},1}) where T
    isempty(v) && throw(ArgumentError("median of an empty array is undefined, $(repr(v))"))
    all(ismissing, v) && return missing
    (eltype(v)<:AbstractFloat || eltype(v)>:AbstractFloat) && any(isnan, v) && return convert(eltype(v), NaN)
    nmis = mapreduce(ismissing, +, v)
    n = length(v) - nmis
    mid = div(1+n,2)
    if isodd(n)
        return middle(partialsort(v,mid))
    else
        m = partialsort(v, mid:mid+1)
        return middle(m[1], m[2])
    end
end

# This can be optimized (using method like median())
function Statistics.quantile(x::Array{T,1},v) where T
    all(ismissing,x) && return missing
    quantile(skipmissing(x),v)
end

# finding k largest in an array with missing values
swap!(x,i,j)=x[i],x[j]=x[j],x[i]

function insert_fixed_sorted!(x, item, ord)
    if ord((x[end]), (item))
        return
    else
        x[end] = item
    end
    i = length(x) - 1
    while i > 0
        if ord((x[i+1]),(x[i]))
            swap!(x,i, i+1)
            i -= 1
        else
            break
        end
    end
end

function k_largest(x::Vector{T}, k::Int) where T
    k < 1 && throw(ArgumentError("k must be greater than 1"))
    k == 1 && return [maximum(identity, x)]
    if k>length(x)
        k = length(x)
    end
    res = Vector{T}(undef,k)
    fill!(res, typemin(T))
    for i in 1:length(x)
        insert_fixed_sorted!(res, x[i], (y1,y2)-> y1 > y2)
    end
    res
end
function k_largest(x::Vector{Union{T,Missing}}, k::Int) where T
    k < 1 && throw(ArgumentError("k must be greater than 1"))
    k == 1 && return [maximum(identity, x)]
    all(ismissing, x) && return missing
    res = Vector{T}(undef,k)
    fill!(res, typemin(T))
    cnt = 0
    for i in 1:length(x)
        if !ismissing(x[i])
            insert_fixed_sorted!(res, x[i], (y1,y2)-> y1 > y2)
            cnt += 1
        end
    end
    if cnt < k
        res[1:cnt]
    else
        res
    end
end

function k_smallest(x::Vector{T}, k::Int) where T
    k < 1 && throw(ArgumentError("k must be greater than 1"))
    k == 1 && return [minimum(identity, x)]
    if k>length(x)
        k = length(x)
    end
    res = Vector{T}(undef,k)
    fill!(res, typemax(T))
    for i in 1:length(x)
        insert_fixed_sorted!(res, x[i], (y1,y2)-> y1 < y2)
    end
    res
end
function k_smallest(x::Vector{Union{T,Missing}}, k::Int) where T
    k < 1 && throw(ArgumentError("k must be greater than 1"))
    k == 1 && return [minimum(identity, x)]
    all(ismissing, x) && return missing
    res = Vector{T}(undef,k)
    fill!(res, typemax(T))
    cnt = 0
    for i in 1:length(x)
        if !ismissing(x[i])
            insert_fixed_sorted!(res, x[i], (y1,y2)-> y1 < y2)
            cnt += 1
        end
    end
    if cnt < k
        res[1:cnt]
    else
        res
    end
end
