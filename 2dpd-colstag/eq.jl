using LinearAlgebra
using Base.Threads

function prepare_pressure_eq_A(dx, dy, sz, gc)
    A = zeros(sz..., 5)
    A[:, :, 1] .= 1

    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            Ae = Aw = 1/(dx^2)
            An = As = 1/(dy^2)
            Ac = - (Ae + Aw + An + As)
            A[i, j, 1] = Ac
            A[i, j, 2] = Ae
            A[i, j, 3] = Aw
            A[i, j, 4] = An
            A[i, j, 5] = As
        end
    end 

    # # upper outer layer, pc - ps = 0
    # @threads for i = gc + 1:sz[1] - gc
    #     j = sz[2] - gc + 1
    #     A[i, j, 1] = 1
    #     A[i, j, 5] = - 1
    # end

    # # lower outer layer, pc - pn = 0
    # @threads for i = gc + 1:sz[1] - gc
    #     j = gc
    #     A[i, j, 1] = 1
    #     A[i, j, 4] = - 1
    # end

    # # right outer layer, pc + pw = 0
    # @threads for j = gc + 1:sz[2] - gc
    #     i = sz[1] - gc + 1
    #     A[i, j, 1] = 1
    #     A[i, j, 3] = - 1
    # end

    # #left outer layer, pc - pe = 0
    # @threads for j = gc + 1:sz[2] - gc
    #     i = gc
    #     A[i, j, 1] = 1
    #     A[i, j, 2] = - 1
    # end

    max_diag = maximum(abs.(A[:, :, 1]))

    A ./= max_diag

    return A, max_diag
end

function init_pressure_eq(P, dx, dy, sz, gc)
    A, max_diag = prepare_pressure_eq_A(dx, dy, sz, gc)
    b = zeros(sz..., P + 1)
    r = zeros(sz..., P + 1)
    return A, b, r, max_diag
end

function cell_resK(A, x, b, i, j)
    xc = x[i, j]
    xe = x[i + 1, j]
    xw = x[i - 1, j]
    xn = x[i, j + 1]
    xs = x[i, j - 1]
    Ac = A[i, j, 1]
    Ae = A[i, j, 2]
    Aw = A[i, j, 3]
    An = A[i, j, 4]
    As = A[i, j, 5]
    r  = b[i, j] - (Ac*xc + Ae*xe + Aw*xw + An*xn + As*xs)
    return r
end

function colored_sor_sweepK!(A, xK, bK, ω, sz, gc, color)
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            if (i + j)%2 == color
                xK[i, j] += ω*cell_resK(A, xK, bK, i, j)/A[i, j, 1]
            end
        end
    end
end

function eq_resK!(A, xK, bK, rK, sz, gc)
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            rK[i, j] = cell_resK(A, xK, bK, i, j)
        end
    end
    mag = norm(rK)
    effective_cell_cnt = (sz[1] - 2*gc + 2)*(sz[2] - 2*gc + 2)
    return mag/sqrt(effective_cell_cnt)
end

function solve_pressure_eq!(A, x, b, r, ω, sz, gc, P, tol, max_it; verbose=false)
    fill!(r, 0)
    it = 0
    for K = 1:P + 1
        xK = @view x[:, :, K]
        bK = @view b[:, :, K]
        rK = @view r[:, :, K]
        while true
            colored_sor_sweepK!(A, xK, bK, ω, sz, gc, 0)
            colored_sor_sweepK!(A, xK, bK, ω, sz, gc, 1)
            err = eq_resK!(A, xK, bK, rK, sz, gc)
            it += 1
            if err <= tol || it >= max_it
                if verbose
                    println("it=$it, err=$err")
                end
                break
            end
        end
    end
end