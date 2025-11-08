using PolyChaos
using DataFrames
using CSV

include("solve.jl")
include("basic_setup.jl")

degree = 5
nsample = degree+1
op = GaussOrthoPoly(degree, Nrec=nsample+1, addQuadrature=true)
x = op.quad.nodes
w = op.quad.weights

println("ulid = $x")
println("weight = $w")