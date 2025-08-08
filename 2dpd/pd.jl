using NLsolve
using LinearAlgebra
using JSON
using Intervals

function intersec_len(interv1, interv2)
    intersec = intersect(interv1, interv2)
    return last(intersec) - first(intersec)
end

function cell_PD_forceK(uc, vc, Umagc, dfuncc, T2, T3, K, P)
    fxc, fyc = 0.0, 0.0
    for J = 1:P + 1, I = 1:P + 1
        a = Umagc[J]*T3[I, J, K]
        fxc += uc[I]*a
        fyc += vc[I]*a
    end
    b = dfuncc/T2[K, K]
    fxc *= b
    fyc *= b
    return fxc, fyc
end

function cell_PD_Umag_guess(uc, vc)
    return sqrt.(uc.^2 + vc.^2)
end

function PD_Umag_init_guess!(u, v, Umag, sz)
    for j = 1:sz[2], i = 1:sz[1]
        Umag[i, j, :] .= cell_PD_Umag_guess(u[i, j, :], v[i, j, :])
    end 
end

function solve_cell_PD_Umag(uc, vc, T3, P; guess = cell_PD_Umag_guess(uc, vc))
    function F!(F, x)
        for K = 1:P + 1
            lsum, rsum = 0.0, 0.0
            for J = 1:P + 1, I = 1:P + 1
                a = T3[I, J, K]
                lsum += x[I]*x[J]*a
                rsum += (uc[I]*uc[J] + vc[I]*vc[J])*a
            end
            F[K] = lsum - rsum
        end
    end

    function J!(J, x)
        fill!(J, 0.0)
        for K = 1:P + 1, L = 1:P + 1
            j = 0.0
            for I = 1:P + 1
                j += x[I]*(T3[L, I, K] + T3[I, L, K])
            end
            J[K, L] = j
        end
    end

    sol = nlsolve(F!, J!, guess, ftol=1e-6)
    if converged(sol)
        return sol.zero
    else
        error("solver for PD coefficients failed to converg. Final guess = $(sol.zero)")
    end
end

function solve_PD_Umag!(u, v, Umag, dfunc, T3, P, sz, gc)
    for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
        if dfunc[i, j] > 1e-6
            uc = u[i, j, :]
            vc = v[i, j, :]
            Umagc_guess = Umag[i, j, :]
            try
                Umagc = solve_cell_PD_Umag(uc, vc, T3, P, guess=Umagc_guess)
                Umag[i, j, :] .= Umagc
            catch
                println("\npd solver error raised at cell ($i $j)")
                rethrow(pd_solver_error)
            end
        else
            Umag[i, j, :] .= 0
        end
    end
end

function prepare_dfunc!(wt_info_json, x, y, dx, dy, dfunc, sz)
    diameter = wt_info_json["diameter"]
    model_diameter = 1.5*diameter
    sigma = model_diameter/6.0
    thick = wt_info_json["thick"]
    C = wt_info_json["C"]
    wt_array_json = wt_info_json["coordinates"]
    wt_count = length(wt_array_json)
    println("PD MODEL INFO")
    println("\tC = $C")
    println("\tturbine thickness = $thick")
    println("\tturbine diameter = $diameter")
    println("\tmodeled turbine diameter = $model_diameter")
    for t = 1:wt_count
        wt_xy = wt_array_json[t]
        println("\tturbine $t at ($(wt_xy[1]), $(wt_xy[2]))")
    end
    for j = 1:sz[2], i = 1:sz[1]
        cell_x_range = Interval(x[i] - 0.5*dx, x[i] + 0.5*dx)
        cell_y_range = Interval(y[j] - 0.5*dy, y[j] + 0.5*dy)
        for t = 1:wt_count
            wt_xy = wt_array_json[t]
            wtx = wt_xy[1]
            wty = wt_xy[2]
            wt_x_range = Interval(wtx - 0.5*thick, wtx + 0.5*thick)
            wt_y_range = Interval(wty - 0.5*model_diameter, wty + 0.5*model_diameter)
            cover = intersec_len(cell_x_range, wt_x_range)*intersec_len(cell_y_range, wt_y_range)
            if cover > 0
                r = abs(y[j] - wty)
                gauss_kernel = exp(- 0.5*(r^2/sigma^2))
                occupancy = cover/(dx*dy)
                dfunc[i, j] = C*gauss_kernel*occupancy
            end
        end
    end
end