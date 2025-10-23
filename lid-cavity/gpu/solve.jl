using CSV
using DataFrames

include("bc.jl")
include("cfd.jl")
include("eq.jl")

nthread_2d = (16, 16)

function time_integral!(
    uold, vold, u, v, divU, ulid,
    A, p, b, r, ω, maxerr, maxit,
    T2, T3, P,
    Re, dx, dt,
    sz, gc;
    maxdiag = 1.0
)
    uold .= u
    vold .= v

    gpu_pseudo_U!(
        uold, vold, u, v, b,
        T2, T3, P, 1.0/Re, dx, dt, maxdiag, sz, gc, 
        nthread_2d
    )
    gpu_solve_eq!(
        A, p, b, r, ω, P, sz, gc, maxerr, maxit,
        nthread_2d
    )
    gpu_pbc!(
        p, P, sz, gc
    )
    gpu_update_U!(
        u, v, p, P, dx, dt, sz, gc,
        nthread_2d
    )
    gpu_Ubc!(
        u, v, ulid, P, sz, gc
    )
    magdivU = gpu_divU!(
        u, v, divU, P, dx, sz, gc, 
        nthread_2d
    )
    return magdivU
end

function get_cell_statistics(v, T2, P)
    var = 0
    for K = 2:P+1
        var += v[K]^2 * T2[K, K]
    end
    mean = v[1]
    return mean, var
end

function write_csv(path, u, v, x, y, P, sz, gc)
    x_coord = zeros(sz...)
    y_coord = zeros(sz...)
    z_coord = zeros(sz...)
    for j = 1:sz[2], i = 1:sz[1]
        x_coord[i, j] = x[i]
        y_coord[i, j] = y[j]
    end

    df = DataFrame(
        x = vec(@view x_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        y = vec(@view y_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        z = vec(@view z_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    )

    unames = [Symbol("u$(K-1)") for K = 1:P + 1]
    vnames = [Symbol("v$(K-1)") for K = 1:P + 1]
    pnames = [Symbol("p$(K-1)") for K = 1:P + 1]

    uhost = Array(u)
    vhost = Array(v)
    phost = Array(p)

    for K = 1:P+1
        df[!, unames[K]] = vec(
            @view uhost[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K]
        )
        df[!, vnames[K]] = vec(
            @view vhost[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K]
        )
        df[!, pnames[K]] = vec(
            @view phost[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K]
        )
    end

    CSV.write(path, df)

    println("written to $path")
end

# checked