using PolyChaos
using DataFrames
using CSV

include("solve.jl")

degree = 5
nrec = 10
op = GaussOrthoPoly(degree, Nrec=nrec, addQuadrature=true)
x = op.quad.nodes
w = op.quad.weights
nsample = length(x)
L=1
N=100
Re=400
T=30
dt=1e-3
μ=1
σ=0.25

println("ulid = $x")
println("weight = $w")