linear_map = nothing

function prepare_pressure_eq_A(dx, dy, sz)
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
    A, max_diag = prepare_pressure_eq_A(dx, dy, sz)
    b = zeros(sz..., P + 1)
    return A, b, max_diag
end

function colored_sor_sweep!(A, x, b, sz, gc, color)
    
end