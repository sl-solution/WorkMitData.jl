using WorkMitData, BenchmarkTools, Random, DFTranspose

df = DataFrame(rand(10^5,100), :auto)
insertcols!(df, 1 , :g=>rand(1:10000, nrow(df)), :rowID=> 1:nrow(df))


df_stack = t_function(df, r"x", [:rowID, :g])

println("stack()")
@btime stack(df, r"x", [:rowID, :g])

println("t_function for stacking")
@btime t_function(df, r"x", [:rowID, :g])
println("df_transpose for stacking")
@btime df_transpose(df, r"x", [:rowID, :g])

println("unstack()")
@btime unstack(df_stack, [:g, :rowID], :_variables_, :_c1)

println("t_function for unstack with id")
@btime t_function(df_stack, :_c1, [:g, :rowID], id = :_variables_)

println("df_transpose for unstack with id")
@btime df_transpose(df_stack, :_c1, [:g, :rowID], id = :_variables_)

println("t_function for unstack without id")
@btime t_function(df_stack, :_c1, [:g, :rowID])

println("df_transpose for unstack without id")
@btime df_transpose(df_stack, :_c1, [:g, :rowID])
