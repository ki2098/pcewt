include("solver.jl")

L=1
N=128
Re=1000
T=30
dt=1e-3

solve(1, Re, L, N, 2, T, dt, "data/test.csv")