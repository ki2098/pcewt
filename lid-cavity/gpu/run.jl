include("solve.jl")
include("pcem.jl")

old_dir = pwd()
cd(@__DIR__)
println("now working in $(@__DIR__)")

T = 20
dt = 1e-3
maxstep::Int = T/dt

cavitySize = 1.0
uLidMean = 1.0
uLidSd = 0.25
N = 100
gc = 2
dx = cavitySize/N
sz = (N+2*gc, N+2*gc)
Re = 400
P = 5

x = [dx*(i - gc - 0.5) - 0.5*cavitySize for i in 1:sz[1]]
y = [dx*(j - gc - 0.5) - 0.5*cavitySize for j in 1:sz[2]]

T2host, T3host = prepare_tensors(P)
T2 = CuArray(T2host)
T3 = CuArray(T3host)

u = CUDA.zeros(Float64, sz..., P+1)
v = CUDA.zeros(Float64, sz..., P+1)
uold = CUDA.zeros(Float64, sz..., P+1)
vold = CUDA.zeros(Float64, sz..., P+1)
divU = CUDA.zeros(Float64, sz..., P+1)
p = CUDA.zeros(Float64, sz..., P+1)
A, b, r, maxdiag = gpu_init_pressure_eq(P, dx, sz, gc, nthread_2d)
ulidhost = [uLidMean, uLidSd, zeros(P-1)...]
ulid = CuArray(ulidhost)

println("cell count = ($(sz[1]) $(sz[1]))")
println("guide cell = $gc")
println("dx = $dx")
println("dt = $dt")
println("Re = $Re")
println("total steps = $maxstep")
println("max diag(A) = $maxdiag")
println("u lid = $ulidhost")
println("P = $P")
println()

gpu_Ubc!(u, v, ulid, P, sz, gc)
for step = 1:maxstep
    magdivU = time_integral!(
        uold, vold, u, v, divU, ulid,
        A, p, b, r, 1.2, 1e-6, 1000,
        T2, T3, P, Re, dx, dt, sz, gc,
        maxdiag=maxdiag
    )
    print("\r step = $step, ||div U|| = $magdivU")
end
println()

mkpath("data")
write_csv("data/result.csv", u, v, x, y, P, sz, gc)

cd(old_dir)