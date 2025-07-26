using Statistics
import LinearSolve
using Printf

include("cfd.jl")
include("eq.jl")
include("bc.jl")

function field_time_integral!(
    uold, vold, u, v, div_U, lid_u,
    p, b,
    Re, dx, dt,
    sz, gc;
    max_A_diag=1.0
)
    uold .= u
    vold .= v
    
    field_pseudo_U!(uold, vold, u, v, 1.0/Re, dx, dt, sz, gc)
    field_pressure_eq_rhs!(u, v, b, dx, dt, max_A_diag, sz, gc)
    linsolve.b = vec(b)
    solution = LinearSolve.solve!(linsolve)
    p .= reshape(solution.u, size(b))
    field_update_U_by_grad_p!(u, v, p, dx, dt, sz, gc)
    apply_U_bc!(u, v, lid_u, sz, gc)
    avg_mag_div_U = field_div_U!(u, v, div_U, dx, sz, gc)

    return avg_mag_div_U
end

function solve(lid_u, cavity_size, Re, dx, T, dt, sz, gc, output_path; max_A_diag=1.0)
    x_coords = [dx*(i - gc - 0.5) - 0.5*cavity_size for i in 1:sz[1]]
    y_coords = [dx*(j - gc - 0.5) - 0.5*cavity_size for j in 1:sz[2]]

    u = zeros(sz)
    v = zeros(sz)
    uold = zeros(sz)
    vold = zeros(sz)
    div_U = zeros(sz)
    p = zeros(sz)
    max_step::Int = T/dt
    apply_U_bc!(u, v, lid_u, sz, gc)
    for step = 1:max_step
        mag_div = field_time_integral!(
            uold, vold, u, v, div_U, lid_u,
            p, b,
            Re, dx, dt,
            sz, gc,
            max_A_diag = max_A_diag
        )
        if !isfinite(mag_div)
            error("invalid value $mag_div occured")
        end
        print("\rstep = $step, mag div = $(round(mag_div, digits=6))")
    end
    println()

    open(output_path, "w") do f
        write(f, "x,y,z,u,v\n")
        for j in gc+1:sz[2]-gc, i in gc+1:sz[1]-gc
            write(f, "$(x_coords[i]),$(y_coords[j]),0,$(u[i,j]),$(v[i,j])\n")
        end
    end
end