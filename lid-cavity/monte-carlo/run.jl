include("solver.jl")

old_dir = pwd()

cd(@__DIR__)
println("now working in $(@__DIR__)")

cavity_size = 1.0
lid_u_mean = 1.0
lid_u_sd = 0.25
n = 100
gc = 2
dx = cavity_size/n
sz = (n + 2*gc, n + 2*gc)
Re = 400

A, b, max_A_diag = init_linear_eq(dx, sz, gc)

T = 20
dt = 1e-3

println("cell count = ($(sz[1]) $(sz[1]))")
println("guide cell = $gc")
println("dx = $dx")
println("dt = $dt")
println("Re = $Re")
println("total steps = $(Int(T/dt))")
println("max A diag = $max_A_diag")
println()
using Distributions
sample_count = 500
X = Normal(lid_u_mean, lid_u_sd)
lid_u = rand(X, sample_count)

try
    rm("data", recursive=true)
catch e
    @warn Exception=e    
end

mkpath("data")
for i = 1:sample_count
    println("running sample $(i), u lid = $(round(lid_u[i], digits=5))")
    solve(
        lid_u[i], cavity_size,
        Re, dx, T, dt,
        sz, gc, "data/result_$(i).csv",
        max_A_diag=max_A_diag
    )
end

using Dates
using JSON
msg_json = Dict("time" => now(), "sample count" => sample_count, "size" => [sz[1]-2*gc, sz[2]-2*gc])
open("sum.json", "w") do io
    JSON.print(io, msg_json)
end
cd(old_dir)