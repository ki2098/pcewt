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

function get_U_var(u, v, normsq, sz, gc)
    u_var = zeros(sz)
    v_var = zeros(sz)
    for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
        cell_u_var, cell_v_var = 0.0, 0.0
        for K = 2:P + 1
            cell_u_var += u[i, j, K]^2 * normsq[K]
            cell_v_var += v[i, j, K]^2 * normsq[K]
        end
        u_var[i, j] = cell_u_var
        v_var[i, j] = cell_v_var
    end
    return u_var, v_var
end