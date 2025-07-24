include("solve.jl")
include("pcem.jl")

cavity_size = 1.0
lid_u_mean = 1.0
lid_u_sd = 1.0
n = 128
gc = 2
dx = cavity_size/n
sz = (n + 2*gc, n + 2*gc)
Re = 1000
P = 5

M = prepare_M(P)

u = zeros(sz..., P + 1)
v = zeros(sz..., P + 1)
uold = zeros(sz..., P + 1)
vold = zeros(sz..., P + 1)
divU = zeros(sz..., P + 1)
p = zeros(sz..., P + 1)

A, b, max_A_diag = init_pressure_eq(P, dx, sz, gc)

T = 10
dt = 1e-3
max_step::Int = T/dt

println("cell count = ($(sz[1]) $(sz[1]))")
println("guide cell = $gc")
println("dx = $dx")
println("dt = $dt")
println("Re = $Re")
println("total steps = $max_step")
println("max A diag = $max_A_diag")
println()
println("non-zero M entries")
show_M(M)

apply_U_bc!(u, v, lid_u_mean, lid_u_sd, P, sz, gc)
# for step = 1:max_step
#     rms_divU = time_integral!(
#         uold, vold, u, v, divU, lid_u_mean, lid_u_sd,
#         p, b,
#         M, P,
#         Re, dx, dt,
#         sz, gc,
#         max_A_diag = max_A_diag
#     )
#     print("\tstep = $step, rms divU = $rms_divU")
# end
# println()
