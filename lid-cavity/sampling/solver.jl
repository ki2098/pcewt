using CSV
using DataFrames
using Printf

include("bc.jl")
include("cfd.jl")
include("eq.jl")

nthread=(16,16)

function gpu_time_integral!(
    u, v, uold, vold, divU, ulid,
    A, p, b, r, maxdiag,
    μ, dx, dt,
    sz, gc
)
    uold .= u
    vold .= v
    gpu_pseudo_U!(
        u, v, uold, vold, b,
        μ, dx, dt, maxdiag,
        sz, gc, nthread
    )
    lsit, lserr = gpu_sor!(
        A, p, b, r,
        1.2, 1e-5, 1000,
        sz, gc, nthread
    )
    gpu_pbc!(
        p, sz, gc
    )
    gpu_update_U!(
        u, v, p,
        dx, dt,
        sz, gc, nthread
    )
    gpu_Ubc!(
        u, v, ulid, sz, gc
    )
    divmag = gpu_divU!(
        u, v, divU, dx,
        sz, gc, nthread
    )
    return lsit, lserr, divmag
end

function write_csv(path, ud, vd, pd, x, y, sz, gc)
    x_coord = zeros(sz...)
    y_coord = zeros(sz...)
    z_coord = zeros(sz...)
    for j = 1:sz[2], i = 1:sz[1]
        x_coord[i, j] = x[i]
        y_coord[i, j] = y[j]
    end
    u = Array(ud)
    v = Array(vd)
    p = Array(pd)

    df = DataFrame(
        x = vec(@view x_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        y = vec(@view y_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        z = vec(@view z_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    )
    df[!, :u] = vec(@view u[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    df[!, :v] = vec(@view v[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    df[!, :p] = vec(@view p[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    CSV.write(path, df)
end

function solve(ulid, Re, cavsize, division, gc, T, dt, outpath)
    dx = cavsize/division
    sz = (division+2*gc, division+2*gc)
    x = [dx*(i-gc-0.5) - 0.5*cavsize for i in 1:sz[1]]
    y = [dx*(j-gc-0.5) - 0.5*cavsize for j in 1:sz[2]]

    u = CUDA.zeros(sz...)
    v = CUDA.zeros(sz...)
    uold = CUDA.zeros(sz...)
    vold = CUDA.zeros(sz...)
    divU = CUDA.zeros(sz...)
    p = CUDA.zeros(sz...)
    A, b, r, maxdiag = gpu_init_pressure_eq(dx, sz, gc, nthread)
    gpu_Ubc!(u, v, ulid, sz, gc)
    maxstep::Int = T/dt

    @printf "sz = %s\n" sz
    @printf "dx = %e\n" dx
    @printf "dt = %e\n" dt
    @printf "total step = %d\n" maxstep
    @printf "Re = %e\n" Re
    @printf "max diag(A) = %e\n" maxdiag

    for step = 1:maxstep
        lsit, lserr, divmag = gpu_time_integral!(
            u, v, uold, vold, divU, ulid,
            A, p, b, r, maxdiag,
            1/Re, dx, dt, sz, gc
        )
        if !isfinite(lserr) || !isfinite(divmag)
            error("convergence failure lserr=$lserr, divmag=$divmag")
        end
        @printf "\rstep=%d ls=(%d %e) div=%e" step lsit lserr divmag
    end
    println()
    write_csv(outpath, u, v, p, x, y, sz, gc)
end