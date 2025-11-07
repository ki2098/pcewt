using CSV
using DataFrames
using Printf

include("bc.jl")
include("cfd.jl")
include("eq.jl")

nthread_2d = (16, 16)

struct Solve
    u
    v
    ut
    vt
    uu
    vv
    div_U
    u_lid
    p
    A
    b
    r
    max_diag
    x
    y
    dx
    dy
    Re
    dt
    max_step
    sz
    gc
end

function info(sol::Solve)
    sz = sol.sz
    println("size = ($(sz[1]) $(sz[2]))")
    gc = sol.gc
    println("gc = $gc")
    Re = sol.Re
    println("Re = $Re")
    maxDiag = sol.max_diag
    println("max diag(A) = $maxDiag")
    uLid = sol.u_lid
    println("u lid = $uLid")
    maxStep = sol.max_step
    println("n step = $maxStep")
end

function init(L, division, u_lid, Re, T, dt)
    gc = 2
    sz = (division + 2*gc, division + 2*gc)
    dx = dy = L/division
    u = CUDA.zeros(sz)
    v = CUDA.zeros(sz)
    ut = CUDA.zeros(sz)
    vt = CUDA.zeros(sz)
    uu = CUDA.zeros(sz)
    vv = CUDA.zeros(sz)
    div_U = CUDA.zeros(sz)
    p = CUDA.zeros(sz)
    A, b, r, max_diag = gpu_init_pressure_eq(dx, dy, sz, gc, nthread_2d)
    x = [(i - gc - 0.5)*dx - 0.5*L for i=1:sz[1]]
    y = [(j - gc - 0.5)*dy - 0.5*L for j=1:sz[2]]
    max_step::Int = T/dt

    gpu_Ubc!(
        u, v, u_lid, sz, gc
    )

    sol = Solve(
        u, v, ut, vt, uu, vv, div_U, u_lid,
        p, A, b, r, max_diag,
        x, y, dx, dy,
        Re, dt, max_step,
        sz, gc
    )

    info(sol)

    return sol
end

function time_integral!(sol::Solve)
    sol.ut .= sol.u
    sol.vt .= sol.v

    gpu_predict_U!(
        sol.ut, sol.vt, sol.u, sol.v, sol.uu, sol.vv,
        sol.dx, sol.dy, sol.dt, 1/sol.Re,
        sol.sz, sol.gc, nthread_2d
    )

    gpu_UUbc!(
        sol.uu, sol.vv, sol.sz, sol.gc
    )

    gpu_pressure_eq_b!(
        sol.uu, sol.vv, sol.b,
        sol.dx, sol.dy, sol.dt, sol.max_diag,
        sol.sz, sol.gc, nthread_2d
    )

    it, err = gpu_sor!(
        sol.A, sol.p, sol.b, sol.r,
        1.5, sol.sz, sol.gc, 1e-6, 1000, nthread_2d
    )

    gpu_pbc!(
        sol.p, sol.sz, sol.gc
    )

    gpu_update_U!(
        sol.u, sol.v, sol.uu, sol.vv, sol.p,
        sol.dx, sol.dy, sol.dt,
        sol.sz, sol.gc, nthread_2d
    )

    gpu_Ubc!(
        sol.u, sol.v, sol.u_lid, sol.sz, sol.gc
    )

    gpu_UUbc!(
        sol.uu, sol.vv, sol.sz, sol.gc
    )

    div_err = gpu_div_U!(
        sol.uu, sol.vv, sol.div_U,
        sol.dx, sol.dy,
        sol.sz, sol.gc, nthread_2d
    )

    return it, err, div_err
end

function write_csv(path::String, sol::Solve)
    u = Array(sol.u)
    v = Array(sol.v)
    p = Array(sol.p)
    div_U = Array(sol.div_U)
    x = sol.x
    y = sol.y
    sz = sol.sz
    gc = sol.gc
    x_coord = zeros(sz)
    y_coord = zeros(sz)
    z_coord = zeros(sz)
    for j = 1:sz[2], i = 1:sz[1]
        x_coord[i, j] = x[i]
        y_coord[i, j] = y[j]
    end

    df = DataFrame(
        x=vec(@view x_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        y=vec(@view y_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        z=vec(@view z_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        u=vec(@view u[gc+1:sz[1]-gc, gc+1:sz[2]-gc]), 
        v=vec(@view v[gc+1:sz[1]-gc, gc+1:sz[2]-gc]), 
        p=vec(@view p[gc+1:sz[1]-gc, gc+1:sz[2]-gc]), 
        div_U=vec(@view div_U[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    )

    CSV.write(path, df)
end

function run_solver(uLid, L, N, Re, T, dt, oPath)
    sol = init(L, N, uLid, Re, T, dt)
    for step = 1:sol.max_step
        lsIt, lsErr, divMag = time_integral!(sol)
        @printf(
            "\rstep=%7i, |div U|=%.5e, LS=(%4i, %.5e)", 
            step, divMag, lsIt, lsErr
        )
        flush(stdout)
    end
    println()
    write_csv(oPath, sol)
end

