function apply_U_bc!(u, v, lid_u_mean, lid_u_sd, P, sz, gc)
    #=
        top boundary
        u0 = lid_u_mean, u1 = lid_u_sd, uK = 0
        vK = 0
    =#
    for i = gc + 1:sz[1] - gc
        j = sz[2] - gc + 1
        bc_u = [lid_u_mean, lid_u_sd, zeros(P + 1 - 2)...]
        u[i, j    , :] = 2*bc_u - u[i, j - 1, :]
        v[i, j    , :] =        - v[i, j - 1, :]
        u[i, j + 1, :] = 2*bc_u - u[i, j - 2, :]
        v[i, j + 1, :] =        - v[i, j - 2, :]
    end 

    #=
        bottom boundary
        uK = 0, vK = 0
    =#
    for i = gc + 1:sz[1] - gc
        j = gc
        u[i, j    , :] = - u[i, j + 1, :]
        v[i, j    , :] = - v[i, j + 1, :]
        u[i, j - 1, :] = - u[i, j + 2, :]
        v[i, j - 1, :] = - v[i, j + 2, :]
    end

    #=
        right boundary
        uK = 0, vK = 0
    =#
    for j = gc + 1:sz[2] - gc
        i = sz[1] - gc + 1
        u[i    , j, :] = - u[i - 1, j, :]
        v[i    , j, :] = - v[i - 1, j, :]
        u[i + 1, j, :] = - u[i - 2, j, :]
        v[i + 1, j, :] = - v[i - 2, j, :]
    end

    #=
        left boundary
        uK = 0, vK = 0
    =#
    for j = gc + 1:sz[2] - gc
        i = gc
        u[i    , j, :] = - u[i + 1, j, :]
        v[i    , j, :] = - v[i + 1, j, :]
        u[i - 1, j, :] = - u[i + 2, j, :]
        v[i - 1, j, :] = - v[i + 2, j, :]
    end 
end