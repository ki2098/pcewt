function left_side_dfdx(fww, fw, fc, dx)
    return (fww - 4*fw + 3*fc)/(2*dx)
end

function apply_Ubc!(u, v, ut, vt, Euin, SDuin, dx, dt, T2, T3, P, sz, gc)
    #=
        @left boundary
        fixed value inlet
        u0 = E[uin], u1 = SD[uin], u2...P = 0
        v0...P = 0
    =#
    for j = gc + 1:sz[2] - gc, i = 1:gc
        u[i, j, 1] = Euin
        u[i, j, 2] = SDuin
        u[i, j, 3:P + 1] .= 0,0
        v[i, j, :] .= 0.0
    end

    #=
        @right boundary
        convective outlet
    =#
    for j = gc + 1:sz[2] - gc, i = sz[1] - gc + 1:sz[1]
        for K = 1:P + 1
            uK_convection = 0.0
            vK_convection = 0.0
            for J = 1:P + 1, I = 1:P + 1
                utI    = ut[i    , j, I]
                utcJ   = ut[i    , j, J]
                utwJ   = ut[i - 1, j, J]
                utwwJ  = ut[i - 2, j, J]
                vtcJ   = vt[i    , j, J]
                vtwJ   = vt[i - 1, j, J]
                vtwwJ  = vt[i - 2, j, J]
                dutJdx = left_side_dfdx(utwwJ, utwJ, utcJ, dx)
                dvtJdx = left_side_dfdx(vtwwJ, vtwJ, vtcJ, dx)
                uK_convection += utI*dutJdx*T3[I, J, K]
                vK_convection += utI*dvtJdx*T3[I, J, K]
            end
            u[i, j, K] = ut[i, j, K] - dt*uK_convection/T2[K, K]
            v[i, j, K] = vt[i, j, K] - dt*vK_convection/T2[K, K]
        end
    end

    #=
        @upper boundary
        u0...P = u[inner]0...P
        v0...P = 0
    =#
    for i = gc + 1:sz[1] - gc, j = sz[2] - gc + 1:sz[2]
        j_mirror = 2*(sz[2] - gc) + 1 - j
        u[i, j, :] .= u[i, j_mirror, :]
        v[i, j, :] .= .- v[i, j_mirror, :]
    end

    #=
        @lower boundary
        u0...P = u[inner]0...P
        v0...P = 0
    =#
    for i = gc + 1:sz[1] - gc, j = 1:gc
        j_mirror = 2*gc + 1 - j
        u[i, j, :] .= u[i, j_mirror, :]
        v[i, j, :] .= .- v[i, j_mirror, :]
    end
end

function apply_UUbc!(uu, vv, Euin, SDuin, sz, gc)
    #=
        @left boundary
        fixed value inlet
        uu0 = E[uin], uu1 = SD[uin], uu2...P = 0
    =#
    for j = gc + 1:sz[2] - gc
        uu[gc, j, 1] = Euin
        uu[gc, j, 2] = SDuin
        uu[gc, j, 3:P + 1] .= 0
    end

    #=
        @upper boundary
        vv0...P = 0
    =#
    for i = gc + 1:sz[1] - gc
        vv[i, sz[2] - gc, :] .= 0
    end

    #=
        @lower boundary
        vv0...P = 0
    =#
    for i = gc + 1:sz[1] - gc
        vv[i, gc, :] .= 0
    end
end
