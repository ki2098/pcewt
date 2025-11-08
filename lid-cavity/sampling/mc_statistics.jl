using JSON
using CSV
using DataFrames

include("mc_setup.jl")
cd(@__DIR__)

nPoint = N*N
u_mean = zeros(nPoint)
v_mean = zeros(nPoint)
u_var = zeros(nPoint)
v_var = zeros(nPoint)
x_coord, y_coord, z_coord = nothing, nothing, nothing

for sampleId = 1:nSample
    filename = "data/mc_$(sampleId).csv"
    df = CSV.read(filename, DataFrame)
    println("read $filename")
    u = df[!, "u"]
    v = df[!, "v"]
    du = u - u_mean
    dv = v - v_mean
    u_mean .+= du ./ sampleId
    v_mean .+= dv ./ sampleId
    du2 = u - u_mean
    dv2 = v - v_mean
    u_var .+= du .* du2
    v_var .+= dv .* dv2
    if sampleId == 1
        global x_coord = df[:, "x"]
        global y_coord = df[:, "y"]
        global z_coord = df[:, "z"]
    end
end
u_var ./= nSample
v_var ./= nSample


sum_df = DataFrame("x"=>x_coord, "y"=>y_coord, "z"=>z_coord, "E[u]"=>u_mean, "Var[u]"=>u_var, "E[v]"=>v_mean, "Var[v]"=>v_var)
CSV.write("data/mc-statistics.csv", sum_df)