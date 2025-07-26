include("cfd.jl")
include("eq.jl")
include("bc.jl")

function time_integral!(
    uold, vold, u, v, divU, lid_u_mean, lid_u_sd,
    p, b,
    M, P,
    Re, dx, dt,
    sz, gc;
    max_A_diag = 1.0
)
    uold .= u
    vold .= v
    
    pseudo_U!(uold, vold, u, v, M, P, 1.0/Re, dx, dt, sz, gc)
    pressure_eq_b!(u, v, b, P, dx, dt, max_A_diag, sz, gc)
    solve_pressure_eq!(p, b, P)
    update_U_by_grad_p!(u, v, p, P, dx, dt, sz, gc)
    apply_U_bc!(u, v, lid_u_mean, lid_u_sd, P, sz, gc)
    rms_divU = div_UK!(u, v, divU, dx, sz, gc)
    return rms_divU
end

function get_var_U(u, v, normsq, sz, gc)
    var_u = zeros(sz)
    var_v = zeros(sz)
    for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
        cell_var_u, cell_var_v = 0.0, 0.0
        for K = 2:P + 1
            cell_var_u += u[i, j, K]^2 * normsq[K]
            cell_var_v += v[i, j, K]^2 * normsq[K]
        end
        var_u[i, j] = cell_var_u
        var_v[i, j] = cell_var_v
    end
    return var_u, var_v
end