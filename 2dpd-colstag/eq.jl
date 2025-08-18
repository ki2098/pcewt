using LinearAlgebra

linear_map = nothing

function prepare_pressure_eq_A(dx, dy, sz, gc)
    cell_count = prod(sz)
    A = zeros(cell_count, 5)
    global linear_map = LinearIndices(Tuple(sz))
    A[:, 1] .= 1

    for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            Ae = Aw = 1/(dx^2)
            An = As = 1/(dy^2)
            Ac = - (Ae + Aw + An + As)
            idc = linear_map[i, j]
            A[idc, 1] = Ac
            A[idc, 2] = Ae
            A[idc, 3] = Aw
            A[idc, 4] = An
            A[idc, 5] = As
        end
    end 

    # upper outer layer, pc - ps = 0
    for i = gc + 1:sz[1] - gc
        j = sz[2] - gc + 1
        idc = linear_map[i, j]
        A[idc, 1] = 1
        A[idc, 5] = - 1
    end

    # lower outer layer, pc - pn = 0
    for i = gc + 1:sz[1] - gc
        j = gc
        idc = linear_map[i, j]
        A[idc, 1] = 1
        A[idc, 4] = - 1
    end

    # right outer layer, pc + pw = 0
    for j = gc + 1:sz[2] - gc
        i = sz[1] - gc + 1
        idc = linear_map[i, j]
        A[idc, 1] = 1
        A[idc, 3] = 1
    end

    #left outer layer, pc - pe = 0
    for j = gc + 1:sz[2] - gc
        i = gc
        idc = linear_map[i, j]
        A[idc, 1] = 1
        A[idc, 2] = - 1
    end

    max_diag = maximum(abs.(A[:, 1]))

    A ./= max_diag

    return A, max_diag
end

function init_pressure_eq(P, dx, dy, sz, gc)
    A, max_diag = prepare_pressure_eq_A(dx, dy, sz, gc)
    b = zeros(sz..., P + 1)
    return A, b, max_diag
end

function cell_res(A, x, b, i, j)
    xc = x[i, j, :]
    xe = x[i + 1, j, :]
    xw = x[i - 1, j, :]
    xn = x[i, j + 1, :]
    xs = x[i, j - 1, :]
    id = linear_map[i, j]
    Ac = A[id, 1]
    Ae = A[id, 2]
    Aw = A[id, 3]
    An = A[id, 4]
    As = A[id, 5]
    r  = b[i, j, :] - (Ac*xc + Ae*xe + Aw*xw + An*xn + As*xs)
    return r
end

function colored_sor_sweep!(A, x, b, ω, sz, gc, color)
    for j = gc:sz[2] - gc - 1
        for i = gc:sz[1] - gc - 1
            if (i + j)%2 == color
                x[i, j, :] += ω*cell_res(A, x, b, i, j)/A[linear_map[i, j], 1]
            end
        end
    end
end

function eq_res!(A, x, b, r, sz, gc)
    for j = gc:sz[2] - gc - 1
        for i = gc:sz[1] - gc - 1
            r[i, j, :] = cell_res(A, x, b, i, j)
        end
    end
    mag = norm(r)
    effective_cell_cnt = (sz[1] - 2*gc + 2)*(sz[2] - 2*gc + 2)
    return mag/sqrt(effective_cell_cnt)
end

function solve_pressure_eq!(A, x, b, ω, sz, gc, tol, max_it)
    fill!(r, 0)
    it = 0
    while true
        colored_sor_sweep!(A, x, b, ω, sz, gc, 0)
        colored_sor_sweep!(A, x, b, ω, sz, gc, 1)
        res_mag = eq_res!(A, x, b, r, sz, gc)
        it += 1
        if res_mag <= tol || it >= max_it
            return it, res_mag
        end
    end
end