function left_side_dfdx(fww, fw, fc, dx)
    return (3*fc - 4*fw + fww)/(2*dx)
end

function apply_U_bc!(u, v, ut, vt, uin_mean, uin_sd, dx, dt, T2, T3, P, sz, gc)
    #=
        outside left boundary
        fixed value inlet
        u0 = uin_mean, u1 = uin_sd, uK = 0
        vK = 0
    =#
    uin = [uin_mean, uin_sd, zeros(P + 1 - 2)...]
    for j = gc + 1:sz[2] - gc
        for i = 1:gc
            u[i, j, :] = uin
            v[i, j, :] .= 0.0
        end
    end

    #=
        outside right boundary
        convective outlet
    =#
    for j = gc + 1:sz[2] - gc
        for i = sz[1] - gc + 1:sz[1]
            for K = 1:P + 1
                convection_uK = 0.0
                convection_vK = 0.0
                for I = 1:P + 1, J = 1:P + 1
                    coefficient = T3[I, J, K]/T2[K, K]
                    utI    = ut[i    , j, I]
                    utcJ   = ut[i    , j, J]
                    utwJ   = ut[i - 1, j, J]
                    utwwJ  = ut[i - 2, j, J]
                    vtcJ   = vt[i    , j, J]
                    vtwJ   = vt[i - 1, j, J]
                    vtwwJ  = vt[i - 2, j, J]
                    dutJdx = left_side_dfdx(utwwJ, utwJ, utcJ, dx)
                    dvtJdx = left_side_dfdx(vtwwJ, vtwJ, vtcJ, dx)
                    convection_uK += utI*dutJdx*coefficient
                    convection_vK += utI*dvtJdx*coefficient
                end
                u[i, j, K] = ut[i, j, K] - dt*convection_uK
                v[i, j, K] = vt[i, j, K] - dt*convection_vK
            end
        end
    end 

    #=
        lower boundary
        slip
        uK = uK[inner]
        vK = 0 
    =#
    for i = gc + 1:sz[1] - gc
        ubc = u[i, gc + 1, :]
        for j = 1:gc
            j_mirror = 2*gc + 1 - j
            u[i, j, :] = 2*ubc - u[i, j_mirror, :]
            v[i, j, :] = - v[i, j_mirror, :]
        end
    end

    #=
        upper boundary
        slip
        uK = uK[inner]
        vK = 0
    =#
    for i = gc + 1:sz[1] - gc
        ubc = u[i, sz[2] - gc, :]
        for j = sz[2] - gc + 1:sz[2]
            j_mirror = 2*(sz[2] - gc) + 1 - j
            u[i, j, :] = 2*ubc - u[i, j_mirror, :]
            v[i, j, :] = - v[i, j_mirror, :]
        end
    end
end