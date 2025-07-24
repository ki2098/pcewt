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