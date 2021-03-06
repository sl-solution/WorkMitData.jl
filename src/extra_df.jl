"""
    std_names!(::DataFrame) renames all columns to standardised form, i.e. they can be call by :name convention.
"""
std_names!(df::DataFrame) = rename!(df, Dict(names(df) .=> replace.(names(df), r"[^a-z_0-9]"i => "_")))


# function _insertion_phase!(x, newval, i)
#     endpos = x[1][i]
#     if endpos == 2
#         x[2][i] = newval
#         x[1][i] += 1
#         return
#     elseif endpos == 3
#         if isless(newval, x[2][i])
#             x[2][i], x[3][i] = newval, x[2][i]
#             x[1][i] += 1
#             return
#         else
#             x[3][i] = newval
#             x[1][i] += 1
#             return
#         end
#     else
#         j = endpos
#         x[j][i] = newval
#         while j > 2
#             if isless(x[j][i],x[j-1][i])
#                 x[j-1][i], x[j][i] = x[j][i], x[j-1][i]
#                 j -= 1
#             else
#                 break
#             end
#         end
#         x[1][i] += 1
#         return
#     end
# end



# function _insertion_op(x, y)
#     for i in 1:length(x[1])
#         _insertion_phase!(x, y[i], i)
#     end
#     return x
# end

# using DataFrames
# df = DataFrame(rand(1000,100), :auto)
# df2 = df[:,:]
# df2 = insertcols!(df2, 1, :cnt => 2)

# @time mapreduce(identity, _insertion_op, eachcol(df2[!, 2:end]), init = eachcol(df2))

# sort(Matrix(df), dims= 2) == Matrix(df2[!,2:end])


# *************************
# function _swap!(x, j, i)
#     _temp = x[j][i]
#     x[j][i] = x[j-1][i]
#     x[j-1][i] = _temp
# end
# function _core_cal!(x, j, i)
#     while j > 2
#         if  isless(x[j][i], x[j-1][i])
#              _swap!(x, j, i)
#             j -= 1
#         else
#             break
#         end
#     end
# end

# function _insertion_phase!(x, newval, i)
#     endpos = Int(x[1][i])
#     j = endpos
#     x[j][i] = newval
#     _core_cal!(x, j, i)
#     x[1][i] += 1
# end



# function _insertion_op!(x, y)
#     for i in 1:length(x[1])
#         _insertion_phase!(x, y[i], i)
#     end
# end

# function _foldl_impl!(op!::OP, v, itr) where {OP}
#     # Unroll the while loop once; if init is known, the call to op may
#     # be evaluated at compile time
#     y = iterate(itr)
#     y === nothing && return
#     op!(v, y[1])
#     while true
#         y = iterate(itr, y[2])
#         y === nothing && return
#         op!(v, y[1])
#     end
# end

# using DataFrames
# df = DataFrame(rand(10^7,5), :auto)
# df2 = df[:,:]
# df2 = insertcols!(df2, 1, :cnt => 2.)

# y1 = [x for x in eachcol(df2)]
# y2 = y1[2:end]

# @time _foldl_impl!(_insertion_op!,y1, y2);
# @time sort(Matrix(df),dims=2);
# df2
