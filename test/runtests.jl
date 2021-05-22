using WorkMitData
using Dates
using Test
using DataFrames, Random, CategoricalArrays, PooledArrays

@testset "WorkMitData.jl" begin

    df=DataFrame(g=[1,1,1,2,2,4,4,4,4],val=[12,32,44,missing,missing,missing,13,1,missing])
    gdf=groupby(df,:g)
    @test all(isequal.(combine(gdf,:val=>(x->mean(x))=>:mval).mval,[mean([12,32,44]),missing,mean([13,1])]))
    @test all(isequal.(isapprox.(combine(gdf,:val=>(x->var(x))=>:varval).varval,[var([12,32,44]),missing,var([13,1])]), [true,missing,true]))
    @test all(isequal.(isapprox.(combine(gdf,:val=>(x->std(x))=>:stdval).stdval,[std([12,32,44]),missing,std([13,1])]), [true,missing,true]))
    @test maximum([missing,missing,missing])===missing

    @test all(isequal.(transform(df,:val=>lead).val_lead,[32,44,missing,missing,missing,13,1,missing,missing]))

    @test all(isequal.(transform(df,:val=>lag).val_lag,[missing,12,32,44,missing,missing,missing,13,1]))

    @test all(ismissing,lead(rand(10),10))
    @test all(ismissing,lag(rand(10),10))
    x=rand(1000)
    y=lead(x,100)
    @test isapprox(sum(y),sum(x[101:end]))


    @test dttodate(0) == Date(1960,1,1)
    @test dttodate(missing) === missing
    @test dttodate(19168) == Date(2012,6,24)

    @test intck(dttodate(0),dttodate(19168)) == -19168

end

const ≅ = isequal
@testset "general usage" begin

    df = DataFrame(x1 = [1,2,3,4], x2 = [1,4,9,16])
    dft =DataFrame(_variables_ =["x1","x2"], _c1 = [1,1], _c2 = [2,4],
                    _c3 = [3,9], _c4 = [4,16])
    @test t_function(df, [:x1, :x2]) == dft

    df = DataFrame(rand(10,5), :auto)
    df2 = t_function(df, r"x")

    @test Matrix(df) == permutedims(Matrix(df2[!,r"_c"]))

    df = DataFrame(rand(10,5), :auto)
    allowmissing!(df)
    df2 = t_function(df, r"x")

    @test t_function(df, r"x") == t_function(df, :)
    @test t_function(df, r"x") == t_function(df, [1,2,3,4,5])
    @test Matrix(df) == permutedims(Matrix(df2[!,2:end]))

    df = DataFrame(rand(10,5), :auto)
    allowmissing!(df)
    df[1, 4] = missing
    df[4, 5] = missing
    df2 = t_function(df, r"x")
    @test Matrix(df) ≅ permutedims(Matrix(df2[!,Not(:_variables_)]))


    df = DataFrames.hcat!(DataFrame(rand(Bool, 15,2),[:b1,:b2]), DataFrame(rand(15,5),:auto))
    df2 = t_function(df, :)
    @test Matrix(df) == permutedims(Matrix(df2[!,Not(:_variables_)]))



    df = DataFrame(foo = ["one", "one", "one", "two", "two","two"],
                    bar = ['A', 'B', 'C', 'A', 'B', 'C'],
                    baz = [1, 2, 3, 4, 5, 6],
                    zoo = ['x', 'y', 'z', 'q', 'w', 't'])
    df2 = t_function(df, :baz, :foo, id = :bar)
    df3 = DataFrame(foo = ["one", "two"], _variables_ = ["baz", "baz"], A = [1, 4], B = [2, 5], C = [3, 6])

    @test t_function(df, :baz, :foo, id = :bar) == t_function(df, [:baz], :foo, id = :bar)
    @test t_function(df, :baz, :foo, id = :bar) == t_function(df, [:baz], [:foo], id = :bar)
    @test t_function(df, :baz, :foo, id = :bar) == t_function(df, :baz, [:foo], id = :bar)
    @test df2 == df3

    df = DataFrame(id = [1, 2, 3, 1], x1 = rand(4), x2 = rand(4))
    @test_throws AssertionError t_function(df, r"x", id = :id)


    df = DataFrame(rand(1:100, 1000, 10), :auto)
    insertcols!(df, 1, :g => repeat(1:100, inner = 10))
    insertcols!(df, 2, :id => repeat(1:10, 100))
    # duplicate and id within the last group
    df[1000, :id] = 1
    @test_throws AssertionError t_function(df, r"x", [:g], id = :id )


    df = DataFrame(rand(1000, 100), :auto)
    mdf = Matrix(df)

    @test describe(t_function(df, r"x"), sum => :sum, cols = r"_c").sum ≈ sum(mdf, dims = 2)

    df = DataFrame([[1, 2], [1.1, 2.0],[1.1, 2.1],[1.1, 2.0]]
                    ,[:person, Symbol("11/2020"), Symbol("12/2020"), Symbol("1/2021")])
    dft = t_function(df, Not(:person), :person,
                        renamerowid = x -> Date(x, dateformat"m/y"),
                        variable_name = "Date",
                         renamecolid = x -> "measurement")
    dftm = DataFrame(person = [1,1,1,2,2,2],
                Date = Date.(repeat(["2020-11-01","2020-12-01","2021-01-01"], 2)),
                measurement = [1.1, 1.1, 1.1, 2.0, 2.1, 2.0])
    @test dft == dftm


    df = DataFrame(rand(100,5),:auto)
    insertcols!(df, 1, :id => repeat(1:20, 5))
    insertcols!(df, 1, :g => repeat(1:5, inner = 20))

    dft = t_function(df, r"x", [:g, :id], variable_name = "variable", renamecolid = x -> "value")
    dfs = sort(stack(df, r"x"), [:g, :id])

    @test dft == dfs


    df = DataFrame(group = repeat(1:3, inner = 2),
                                 b = repeat(1:2, inner = 3),
                                 c = repeat(1:1, inner = 6),
                                 d = repeat(1:6, inner = 1),
                                 e = string.('a':'f'))
    allowmissing!(df)
    df[2, :b] = missing
    df[4, :d] = missing
    df2 = t_function(df, 2:4, [:group], id = :e, default_fill = 0)

    df3 = DataFrame(group = repeat(1:3, inner = 3),
                    _variables_ = string.(repeat('b':'d', 3)),
                    a = [1,1,1,0,0,0,0,0,0],
                    b = [missing,1,2,0,0,0,0,0,0],
                    c = [0,0,0, 1,1,3, 0,0,0],
                    d = [0,0,0, 2,1,missing, 0,0,0],
                    e = [0,0,0,0,0,0,2,1,5],
                    f = [0,0,0,0,0,0,2,1,6] )
    @test df2 ≅ df3

    df = DataFrame(id = ["r3", "r1", "r2" , "r4"], x1 = [1,2,3,4], x2 = [1,4,9,16])
    df2 = t_function(df, [:x1,:x2], id = :id)
    df3 = DataFrame([["x1", "x2"],
                        [1, 1],
                        [2, 4],
                        [3, 9],
                        [4, 16]],
                        [:_variables_, :r3, :r1, :r2, :r4])
    @test df3 == df2

    pop = DataFrame(country = ["c1","c1","c2","c2","c3","c3"],
                            sex = ["male", "female", "male", "female", "male", "female"],
                            pop_2000 = [100, 120, 150, 155, 170, 190],
                            pop_2010 = [110, 120, 155, 160, 178, 200],
                            pop_2020 = [115, 130, 161, 165, 180, 203])

    popt = t_function(pop, r"pop_", :country,
                            id = :sex, variable_name = "year",
                            renamerowid = x -> match(r"[0-9]+",x).match, renamecolid = x -> x * "_pop")
    poptm = DataFrame([ ["c1", "c1", "c1", "c2", "c2", "c2", "c3", "c3", "c3"],
            SubString{String}["2000", "2010", "2020", "2000", "2010", "2020", "2000", "2010", "2020"],
            Union{Missing, Int64}[100, 110, 115, 150, 155, 161, 170, 178, 180],
            Union{Missing, Int64}[120, 120, 130, 155, 160, 165, 190, 200, 203]],
            [:country,:year,:male_pop,:female_pop])

    @test popt == poptm
    popt = t_function(pop, r"pop", r"cou", id = r"sex",  variable_name = "year",
                            renamerowid = x -> match(r"[0-9]+",x).match, renamecolid = x -> x * "_pop")
    @test popt == poptm
    pop.country = PooledArray(pop.country)
    popt = t_function(pop, r"pop", r"cou", id = r"sex",  variable_name = "year",
                            renamerowid = x -> match(r"[0-9]+",x).match, renamecolid = x -> x * "_pop")
    @test popt.country == PooledArray(["c1", "c1", "c1", "c2", "c2", "c2", "c3", "c3", "c3"])


    df =  DataFrame(region = repeat(["North","North","South","South"],2),
                 fuel_type = repeat(["gas","coal"],4),
                 load = [.1,.2,.5,.1,6.,4.3,.1,6.],
                 time = [1,1,1,1,2,2,2,2],
                 )

    df2 = t_function(df, :load, :time, id = 1:2)
    df3 = DataFrame([[1, 2],
         ["load", "load"],
         Union{Missing, Float64}[0.1, 6.0],
         Union{Missing, Float64}[0.2, 4.3],
         Union{Missing, Float64}[0.5, 0.1],
         Union{Missing, Float64}[0.1, 6.0]], [:time,:_variables_,Symbol("(\"North\", \"gas\")"),Symbol("(\"North\", \"coal\")"),Symbol("(\"South\", \"gas\")"),Symbol("(\"South\", \"coal\")")])

     @test df2 == df3
     df = DataFrame(A_2018=1:4, A_2019=5:8, B_2017=9:12,
                             B_2018=9:12, B_2019 = [missing,13,14,15],
                              ID = [1,2,3,4])
      f(x) =  match(r"[0-9]+",x).match
      dfA = t_function(df, r"A", :ID, renamerowid = f, variable_name = "Year", renamecolid = x->"A");
      dfB = t_function(df, r"B", :ID, renamerowid = f, variable_name = "Year", renamecolid = x->"B");
      df2 = outerjoin(dfA, dfB, on = [:ID, :Year])
      df3 = DataFrame([[1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4],
                 SubString{String}["2018", "2019", "2018", "2019", "2018", "2019", "2018", "2019", "2017", "2017", "2017", "2017"],
                 Union{Missing, Int64}[1, 5, 2, 6, 3, 7, 4, 8, missing, missing, missing, missing],
                 Union{Missing, Int64}[9, missing, 10, 13, 11, 14, 12, 15, 9, 10, 11, 12]], [:ID,:Year,:A,:B])
        @test df2 ≅ df3
        df = DataFrame(paddockId= [0, 0, 1, 1, 2, 2],
                                color= ["red", "blue", "red", "blue", "red", "blue"],
                                count= [3, 4, 3, 4, 3, 4],
                                weight= [0.2, 0.3, 0.2, 0.3, 0.2, 0.2])
        df2 = t_function( t_function(df, [:count,:weight],[:paddockId,:color]),
                             :_c1, :paddockId, id = 2:3)
         df3 = DataFrame([[0, 1, 2],
                         ["_c1", "_c1", "_c1"],
                         Union{Missing, Float64}[3.0, 3.0, 3.0],
                         Union{Missing, Float64}[0.2, 0.2, 0.2],
                         Union{Missing, Float64}[4.0, 4.0, 4.0],
                         Union{Missing, Float64}[0.3, 0.3, 0.2]],[ :paddockId,:_variables_,Symbol("(\"red\", \"count\")"),Symbol("(\"red\", \"weight\")"),Symbol("(\"blue\", \"count\")"),Symbol("(\"blue\", \"weight\")")])

         @test df2 == df3

        df = DataFrame(x1 = [9,2,8,6,8], x2 = [8,1,6,2,3], x3 = [6,5,3,10,8])
        df2 = t_function(df, r"x", renamerowid = x -> match(r"[0-9]+",x).match,renamecolid = x -> "_column_" * string(x))
        df3 = DataFrame([SubString{String}["1", "2", "3"],
                         [9, 8, 6],
                         [2, 1, 5],
                         [8, 6, 3],
                         [6, 2, 10],
                         [8, 3, 8]],[:_variables_,:_column_1,:_column_2,:_column_3,:_column_4,:_column_5])
         @test df2 == df3
         df = DataFrame(a=["x", "y"], b=[1, "two"], c=[3, 4], d=[true, false])
         df2 = t_function(df, [:b, :c, :d], id = :a, variable_name = "new_col")
         df3 = DataFrame([["b", "c", "d"],
                         Any[1, 3, true],
                         Any["two", 4, false]],[:new_col,:x,:y])
         @test df2 == df3
         df = DataFrame(rand(1:100,100,4),:auto)
         DataFrames.hcat!(df, DataFrame(rand(100,3),:auto), makeunique = true)
         insertcols!(df,1, :id=>Symbol.(100:-1:1))

         df2 = t_function(df, r"x", id = :id, variable_name = "id")
         df3 = permutedims(df, :id)

         @test df2 == df3

end


@testset "Outputs - from DataFrames test set" begin
    df = DataFrame(Fish = CategoricalArray{Union{String, Missing}}(["Bob", "Bob", "Batman", "Batman"]),
                   Key = CategoricalArray{Union{String, Missing}}(["Mass", "Color", "Mass", "Color"]),
                   Value = Union{String, Missing}["12 g", "Red", "18 g", "Grey"])
    levels!(df[!, 1], ["XXX", "Bob", "Batman"])
    levels!(df[!, 2], ["YYY", "Color", "Mass"])
#     Not sure if it is relevant, however, we are doing it here
    df2 = t_function(df, [:Value], [:Fish], id = :Key)
    @test levels(df[!, 1]) == ["XXX", "Bob", "Batman"] # make sure we did not mess df[!, 1] levels
    @test levels(df[!, 2]) == ["YYY", "Color", "Mass"] # make sure we did not mess df[!, 2] levels
#     duplicates
    @test_throws AssertionError t_function(df, [:Value], id = :Key)


    df = DataFrame(Fish = CategoricalArray{Union{String, Missing}}(["Bob", "Bob", "Batman", "Batman"]),
                   Key = CategoricalArray{Union{String, Missing}}(["Mass", "Color", "Mass", "Color"]),
                   Value = Union{String, Missing}["12 g", "Red", "18 g", "Grey"])
    levels!(df[!, 1], ["XXX", "Bob", "Batman"])
    levels!(df[!, 2], ["YYY", "Color", "Mass"])
    df2 = t_function(df, [:Value], [:Fish], id = :Key, renamecolid=x->string("_", uppercase(string(x)), "_"))
    df4 = DataFrame(Fish = Union{String, Missing}["Bob", "Batman"],
                    _variables_ = String["Value", "Value"],
                    _COLOR_ = Union{String, Missing}["Red", "Grey"],
                    _MASS_ = Union{String, Missing}["12 g", "18 g"])
    @test df2 == df4
    # without categorical array
    df = DataFrame(Fish = ["Bob", "Bob", "Batman", "Batman"],
                   Key = ["Mass", "Color", "Mass", "Color"],
                   Value = ["12 g", "Red", "18 g", "Grey"])
    df2 = t_function(df, [:Value], [:Fish], id = :Key)
    df4 = DataFrame(Fish = ["Bob", "Batman"],
                    _variables_ = ["Value", "Value"],
                    Mass = ["12 g", "18 g"],
                    Color = ["Red", "Grey"])
    @test df2 ≅ df4
    @test typeof(df2[!, :Fish]) <: Vector{String}
    #Make sure df_transpose works with missing values at the start of the value column
    allowmissing!(df, :Value)
    df[1, :Value] = missing
    df2 = t_function(df, [:Value], [:Fish], id =  :Key)
    #This changes the expected result
    allowmissing!(df4, :Mass)
    df4[1, :Mass] = missing
    @test df2 ≅ df4


    # test missing value in grouping variable
    mdf = DataFrame(RowID = 1:4, id=[missing, 1, 2, 3], a=1:4, b=1:4)
    @test select(t_function(t_function(mdf, [:a,:b], [:RowID, :id]), [:_c1], [:RowID, :id], id = :_variables_), :RowID, :id, :a, :b) ≅ mdf
    @test select(t_function(t_function(mdf, Not(1,2), [:RowID, :id]), [:_c1], [:RowID, :id], id = :_variables_), :RowID, :id, :a, :b) ≅ mdf

    # test more than one grouping column
    wide = DataFrame(id = 1:12,
                     a  = repeat([1:3;], inner = [4]),
                     b  = repeat([1:4;], inner = [3]),
                     c  = randn(12),
                     d  = randn(12))
    w2 = wide[:, [1, 2, 4, 5]]
    rename!(w2, [:id, :a, :_C_, :_D_])
    long = t_function(wide, [:c, :d], [:id, :a, :b])
    wide3 = t_function(long, [:_c1], [:id, :a, :b], id = :_variables_)
    @test select(wide3, Not(:_variables_)) == wide
    df = DataFrame([repeat(1:2, inner=4), repeat('a':'d', outer=2), collect(1:8)],
                       [:id, :variable, :value])

    udf = t_function(df, [:value], [:id], id = :variable)
    @test select(udf, Not(:_variables_)) == DataFrame([Union{Int, Missing}[1, 2], Union{Int, Missing}[1, 5],
                                Union{Int, Missing}[2, 6], Union{Int, Missing}[3, 7],
                                Union{Int, Missing}[4, 8]], [:id, :a, :b, :c, :d])

    @test isa(udf[!, 1], Vector{Int})
    @test isa(udf[!,:a], Vector{Union{Int,Missing}})
    @test isa(udf[!,:b], Vector{Union{Int,Missing}})
    @test isa(udf[!,:c], Vector{Union{Int,Missing}})
    @test isa(udf[!,:d], Vector{Union{Int,Missing}})
    df = DataFrame([categorical(repeat(1:2, inner=4)),
                           categorical(repeat('a':'d', outer=2)), categorical(1:8)],
                       [:id, :variable, :value])

    udf = t_function(df, [:value], [:id], id = :variable)
    @test isa(udf[!, 1], CategoricalArray{Int64,1,UInt32})
    @test isa(udf[!, :a], Vector{Union{Missing, CategoricalValue{Int64, UInt32}}})
    @test isa(udf[!, :b], Vector{Union{Missing, CategoricalValue{Int64, UInt32}}})
    @test isa(udf[!, :c], Vector{Union{Missing, CategoricalValue{Int64, UInt32}}})
    @test isa(udf[!, :d], Vector{Union{Missing, CategoricalValue{Int64, UInt32}}})


    df1 = DataFrame(a=["x", "y"], b=rand(2), c=[1, 2], d=rand(Bool, 2))

    @test_throws MethodError t_function(df1)
    @test_throws ArgumentError t_function(df1, :bar)

    df1_pd = t_function(df1, 2:4, id = 1)
    @test size(df1_pd, 1) == ncol(df1) - 1
    @test size(df1_pd, 2) == nrow(df1) + 1
    @test names(df1_pd) == ["_variables_", "x", "y"]
    @test names(t_function(df1, 2:4, id = 1, variable_name = "foo")) == ["foo", "x", "y"]

end
@testset "byrow tests" begin
    df = DataFrame(g = [1, 1, 1, 2, 2],
                      x1_int = [0, 0, 1, missing, 2],
                      x2_int = [3, 2, 1, 3, -2],
                      x1_float = [1.2, missing, -1.0, 2.3, 10],
                      x2_float = [missing, missing, 3.0, missing, missing],
                      x3_float = [missing, missing, -1.4, 3.0, -100.0])

    q1 = byrow(sum, df)
    @test isequal(q1, [5.2,3.0,3.6,10.3,-88.0])

    q2 = byrow(sum, df, r"x")
    @test isequal(q2, [4.2,2.0,2.6,8.3,-90.0])

    q3 = byrow(mean, df, r"_float")
    @test isequal(q3, [1.2,missing,0.20000000000000004,2.65,-45.0])

    q4 = byrow(maximum, df, 2:6)
    @test isequal(q4, [3.0,2.0,3.0,3.0,10.0])

    q5 = byrow(maximum, df, 2:6, by = abs)
    @test isequal(q5, [3.0,2.0,3.0,3.0,100.0])

    q6 = byrow(var, df, r"_float")
    @test isequal(q6, [0.0,missing,5.92,0.24499999999999922,6050.0])

    q7 = byrow(count, df, by = !ismissing)
    @test isequal(q7, [4,3,6,4,5])

    q8 = byrow(nunique, df, r"int")
    @test q8 == [2,2,1,2,2]

    q9 = byrow(nunique, df)
    @test q9 == [5,4,4,4,5]

    q10 = byrow(nunique, df, by = abs)
    @test q10 == [5,4,3,4,4]

    q11 = byrow(nunique, df, count_missing = false)
    @test q11 == [4,3,4,3,4]
end

