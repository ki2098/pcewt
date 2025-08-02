using NLsolve
using LinearAlgebra
using JSON
using Intervals

function intersec_len(interv1, interv2)
    intersec = intersect(interv1, interv2)
    return last(intersec) - first(intersec)
end

function cell_PD_forceK(uc, vc, Umagc, dfuncc, T2, T3, K, P)
    fx, fy = 0.0, 0.0
    for I = 1:P + 1, J = 1:P + 1
        ef = Umagc[J]*T3[I, J, K]
        fx += uc[I]*ef
        fy += vc[I]*ef
    end
    co = dfuncc/T2[K, K]
    fx *= co
    fy *= co
    return fx, fy
end

function cell_PD_Umag_init_guess(uc, vc)
    return sqrt.(uc.^2 + vc.^2)
end

function PD_Umag_init_guess!(u, v, Umag, sz)
    for i = 1:sz[1], j = 1:sz[2]
        Umag[i, j, :] = cell_PD_Umag_init_guess(u[i, j, :], v[i, j, :])
    end
end

function solve_cell_PD_Umag(uc, vc, Umagc, T3, P)
    function F!(r, x)
        for K = 1:P + 1
            lsum, rsum = 0.0, 0.0
            for I = 1:P + 1, J = 1:P + 1
                lsum += x[I]*x[J]*T3[I, J, K]
                rsum += (uc[I]*uc[J] + vc[I]*vc[J])*T3[I, J, K]
            end
            r[K] = lsum - rsum
        end
    end

    function J!(J, x)
        fill!(J, 0.0)
        for K = 1:P + 1, L = 1:P + 1
            entry = 0.0
            for I = 1:P + 1
                entry += x[I]*(T3[L, I, K] + T3[I, L, K])
            end
            J[K, L] = entry
        end
    end

    sol = nlsolve(F!, J!, Umagc, ftol=(P + 1)*1e-6)
    if converged(sol)
        return sol.zero
    else
        error("solver for PD coefficients failed to converge\nfinal guess = $(sol.zero)\n")
    end
end

function solve_PD_Umag!(u, v, Umag, T3, P, sz, gc)
    for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
        uc = u[i, j, :]
        vc = v[i, j, :]
        Umagc = Umag[i, j, :]
        Umagc_new = solve_cell_PD_Umag(uc, vc, Umagc, T3, P)
        Umag[i, j, :] = Umagc_new
    end
end

function prepare_dfunc!(wt_array_json, x, y, dx, dfunc, sz)
    wt_count = length(wt_array_json)
    for i = 1:sz[1], j = 1:sz[2]
        cell_x_range = Interval(x[i] - 0.5*dx, x[i] + 0.5*dx)
        cell_y_range = Interval(y[j] - 0.5*dx, y[j] + 0.5*dx)
        for t = 1:wt_count
            wt_json = wt_array_json[t]
            wt_xy = wt_json["position"]
            wtx = wt_xy[1]
            wty = wt_xy[2]
            diamater = 1.5*wt_json["diameter"]
            thick = wt_json["thick"]
            wt_x_range = Interval(wtx - 0.5*thick, wtx + 0.5*thick)
            wt_y_range = Interval(wty - 0.5*diamater, wty + 0.5*diamater)
            cover = intersec_len(cell_x_range, wt_x_range)*intersec_len(cell_y_range, wt_y_range)
            if cover > 0
                sigma = diamater/6.0
                dist = abs(y[j] - wty)
                gauss_kernel = exp(- 0.5*(dist/sigma)^2)
                C = wt_json["C"]
                occupancy = cover/(dx^2)
                dfunc[i, j] = C*gauss_kernel*occupancy
            end
        end
    end
end