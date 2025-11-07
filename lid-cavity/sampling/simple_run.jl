include("solver.jl")

L=1
N=100
Re=400
T=30
dt=1e-3

solve(1, Re, L, N, 2, T, dt, "data/test.csv")