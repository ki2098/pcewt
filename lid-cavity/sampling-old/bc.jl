function apply_U_bc!(u, v, lid_u, sz, gc)
    #=
        upper outer layer
        uc = 2*lid_u - us, un = 2*lid_u - uss
        vc = - vs, vn = - vss
    =#
    for i = gc + 1:sz[1] - gc
        j = sz[2] - gc + 1
        u[i, j    ] = 2*lid_u - u[i, j - 1]
        v[i, j    ] =         - v[i, j - 1]
        u[i, j + 1] = 2*lid_u - u[i, j - 2]
        v[i, j + 1] =         - v[i, j - 2]
    end

    #=
        lower outer layer
        uc = - un, us = - unn
        vc = - vn, vs = - vnn
    =#
    for i = gc + 1:sz[1] - gc
        j = gc
        u[i, j    ] = - u[i, j + 1]
        v[i, j    ] = - v[i, j + 1]
        u[i, j - 1] = - u[i, j + 2]
        v[i, j - 1] = - v[i, j + 2]
    end

    #=
        right outer layer
        uc = - uw, ue = - uww
        vc = - vw, ve = - vww
    =#
    for j = gc + 1:sz[2] - gc
        i = sz[1] - gc + 1
        u[i    , j] = - u[i - 1, j]
        v[i    , j] = - v[i - 1, j]
        u[i + 1, j] = - u[i - 2, j]
        v[i + 1, j] = - v[i - 2, j]
    end

    #=
        left outer layer
        uc = - ue, uw = - uee
        vc = - ve, vw = - vee
    =#
    for j = gc + 1:sz[2] - gc
        i = gc
        u[i    , j] = - u[i + 1, j]
        v[i    , j] = - v[i + 1, j]
        u[i - 1, j] = - u[i + 2, j]
        v[i - 1, j] = - v[i + 2, j]
    end
end