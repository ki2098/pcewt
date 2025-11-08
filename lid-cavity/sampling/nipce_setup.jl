using PolyChaos
using DataFrames
using CSV

include("solve.jl")
include("basic_setup.jl")

degree = 5
nrec = 10
op = GaussOrthoPoly(degree, Nrec=nrec, addQuadrature=true)
x = op.quad.nodes
w = op.quad.weights
nsample = length(x)

println("ulid = $x")
println("weight = $w")