using PolyChaos
using DataFrames
using CSV

include("solve.jl")

L=1
N=100
T=30
dt=1e-3

degree=5
nsample=degree+1

op1 = GaussOrthoPoly(degree, Nrec=nsample+1, addQuadrature=true)
op2 = Normal01OrthoPoly(degree, Nrec=nsample+1, addQuadrature=true)

ξ1 = op1.quad.nodes
w1 = op1.quad.weights
ξ2 = op2.quad.nodes
w2 = op2.quad.weights
τ1 = computeSP2(op1)
τ2 = computeSP2(op2)

mop = MultiOrthoPoly([op1, op2], degree)
idxMap = mop.ind
M = mop.dim