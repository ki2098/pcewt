using LinearAlgebra
import KrylovKit
import IterativeSolvers
import LinearSolve
using SparseArrays

Pl = nothing
println("$(@__FILE__) is used for linear solver")

function prepare_pressure_eq_A(dx, dy, sz, gc)
    cell_count = prod(sz)
    A = spdiagm(0 => ones(cell_count))
    map_id = LinearIndices(Tuple(sz))

    for i = gc + 1:sz[1] - gc
        for j = gc + 1:sz[2] - gc
            if i == gc + 1
                Aw = 0
            else
                Aw = 1/(dx^2)
            end

            if i == sz[1] - gc
                Ae = 0
            else
                Ae = 1/(dx^2)
            end

            if j == gc + 1
                As = 0
            else
                As = 1/(dy^2)
            end

            if j == sz[2] - gc
                An = 0
            else
                An = 1/(dy^2)
            end

            Ac = - (Ae + Aw + An + As)

            idc = map_id[i ,j]
            ide = map_id[i + 1, j]
            idw = map_id[i - 1, j]
            idn = map_id[i, j + 1]
            ids = map_id[i, j - 1]
            A[idc, idc] = Ac
            A[idc, ide] = Ae
            A[idc, idw] = Aw
            A[idc, idn] = An
            A[idc, ids] = As
        end
    end

    for i = gc + 1:sz[1] - gc
        j = sz[2] - gc + 1
        idc = map_id[i, j]
        ids = map_id[i, j - 1]
        A[idc, idc] = 1
        A[idc, ids] = - 1
    end

    for i = gc + 1:sz[1] - gc
        j = gc
        idc = map_id[i, j]
        idn = map_id[i, j + 1]
        A[idc, idc] = 1
        A[idc, idn] = - 1
    end

    for j = gc + 1:sz[2] - gc
        i = sz[1] - gc + 1
        idc = map_id[i, j]
        idw = map_id[i - 1, j]
        A[idc, idc] = 1
        A[idc, idw] = - 1
    end

    for j = gc + 1:sz[2] - gc
        i = gc
        idc = map_id[i, j]
        ide = map_id[i + 1, j]
        A[idc, idc] = 1
        A[idc, ide] = - 1
    end

    max_diag = maximum(abs.(diag(A)))

    A ./= max_diag

    global Pl = Diagonal(A)

    return A, max_diag
end

function init_pressure_eq(P, dx, dy, sz, gc)
    A, max_diag = prepare_pressure_eq_A(dx, dy, sz, gc)
    b = zeros(sz..., P + 1)
    return A, b, nothing, max_diag
end

function solve_pressure_eq!(A, x, b, r, ω, sz, gc, P, tol, max_it; verbose=false)
    for K = 1:P + 1
        # prob = LinearSolve.LinearProblem(A, vec(b[:, :, K]), u0=vec(x[:, :, K]))
        # sol = LinearSolve.solve(prob, LinearSolve.KrylovJL_GMRES(), Pl=Pl, abstol=tol*prod(sz), maxiters=max_it, verbose=verbose)
        # x[:, :, K] .= reshape(sol.u, sz)

        IterativeSolvers.gmres!(
            vec(@view x[:, :, K]),
            A,
            vec(@view b[:, :, K]),
            Pl = Pl,
            maxiter = 2000,
            abstol = tol*prod(sz)
        )

        # sol, info = KrylovKit.linsolve(
        #     A, vec(@view b[:, :, K]), vec(@view x[:, :, K]),
        #     tol = tol*prod(sz)
        # )
        # x[:, :, K] .= reshape(sol, sz)
    end
end
