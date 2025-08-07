
function utopia_convection(fww, fw, fc, fe, fee, uW, uE, uc, dx)
    xE = uE*(fw - 27*fc + 27*fe - fee)/(24*dx)
    xW = uW*(fww - 27*fw + 27*fc - fe)/(24*dx)
    xA = abs(uc)*(fww - 4*fw + 6*fc - 4*fe + fee)/(12*dx)
    return 0.5*(xE + xW) + xA
end

function cell_convectionK(uc, uE, uW, vc, vN, vS, f, T2, T3, K, P, dx, dy, i, j)
    convection = 0.0
    for J = 1:P + 1, I = 1:P + 1
        fJ = @view f[:, :, J]
        ucI = uc[I]
        uEI = uE[I]
        uWI = uW[I]
        vcI = vc[I]
        vNI = vN[I]
        vSI = vS[I]
        fcJ  = fJ[i, j]
        feJ  = fJ[i + 1, j]
        feeJ = fJ[i + 2, j]
        fwJ  = fJ[i - 1, j]
        fwwJ = fJ[i - 2, j]
        fnJ  = fJ[i, j + 1]
        fnnJ = fJ[i, j + 2]
        fsJ  = fJ[i, j - 1]
        fssJ = fJ[i, j - 2]
        convection += (
            utopia_convection(fwwJ, fwJ, fcJ, feJ, feeJ, uWI, uEI, ucI, dx)
        +   utopia_convection(fssJ, fsJ, fcJ, fnJ, fnnJ, vSI, vNI, vcI, dy)
        )*T3[I, J, K]
    end
    return convection/T2[K, K]
end

function cell_diffusionK(fK, viscosity, dx, dy, i, j)
    fcK = fK[i, j]
    feK = fK[i + 1, j]
    fwK = fK[i - 1, j]
    fnK = fK[i, j + 1]
    fsK = fK[i, j - 1]
    return viscosity*(
        (feK - 2*fcK + fwK)/(dx^2)
    +   (fnK - 2*fcK + fsK)/(dy^2)
    )
end

function pseudo_U!(ut, vt, u, v, uu, vv, Umag, dfunc, T2, T3, P, viscosity, dx, dy, dt, sz, gc)
    for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
        uc = ut[i    , j, :]
        uE = uu[i    , j, :]
        uW = uu[i - 1, j, :]
        vc = vt[i, j    , :]
        vN = vv[i, j    , :]
        vS = vv[i, j - 1, :]
        Umagc = Umag[i, j, :]
        dfuncc = dfunc[i, j]
        for K = 1:P + 1
            fxc, fyc = cell_PD_forceK(
                uc, vc, Umagc, dfuncc, T2, T3, K, P
            )
            uK_convection = cell_convectionK(
                uc, uE, uW, vc, vN, vS, ut, T2, T3, K, P, dx, dy, i, j
            )
            uK_diffusion = cell_diffusionK(
                (@view ut[:, :, K]), viscosity, dx, dy, i, j
            )
            vK_convection = cell_convectionK(
                uc, uE, uW, vc, vN, vS, vt, T2, T3, K, P, dx, dy, i, j
            )
            vK_diffusion = cell_diffusionK(
                (@view vt[:, :, K]), viscosity, dx, dy, i, j
            )
            u[i, j, K] = uc[K] + dt*(- uK_convection + uK_diffusion - fxc)
            v[i, j, K] = vc[K] + dt*(- vK_convection + vK_diffusion - fyc)
        end
    end
end

function interpolate_UU!(u, v, uu, vv, sz, gc)
    for K = 1:P + 1
        uK = @view u[:, :, K]
        vK = @view v[:, :, K]
        uuK = @view uu[:, :, K]
        vvK = @view vv[:, :, K]
        #=
            uu
            @upper  no
            @lower  no
            @inlet  no
            @outlet yes
        =#
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            uuK[i, j] = 0.5*(uK[i, j] + uK[i + 1, j])
        end
        #=
            vv
            @upper  no
            @lower  no
            @inlet  no
            @outlet no
        =#
        for j = gc + 1:sz[2] - gc - 1, i = gc + 1:sz[1] - gc
            vvK[i, j] = 0.5*(vK[i, j] + vK[i, j + 1])
        end
    end
end

function cell_divUK(uuK, vvK, dx, dy, i, j)
    uEK = uuK[i    , j]
    uWK = uuK[i - 1, j]
    vNK = vvK[i, j    ]
    vSK = vvK[i, j - 1]
    return (uEK - uWK)/(2*dx) + (vNK - vSK)/(2*dy)
end

function pressure_eq_b!(uu, vv, b, P, dx, dy, dt, max_diag, sz, gc)
    for K = 1:P + 1
        uuK = @view uu[:, :, K]
        vvK = @view vv[:, :, K]
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            b[i, j, K] = cell_divUK(uuK, vvK, dx, dy, i, j)/(dt*max_diag)
        end
    end
end

function cell_gradpK(pK, dx, dy, i, j)
    peK = pK[i + 1, j]
    pwK = pK[i - 1, j]
    pnK = pK[i, j + 1]
    psK = pK[i, j - 1]
    return (peK - pwK)/(2*dx), (pnK - psK)/(2*dy)
end

function update_U_by_gradp!(u, v, uu, vv, p, P, dx, dy, dt, sz, gc)
    for K = 1:P + 1
        pK = @view p[:, :, K]
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            dpdx, dpdy = cell_gradpK(pK, dx, dy, i, j)
            u[i, j, K] -= dt*dpdx
            v[i, j, K] -= dt*dpdy
        end
        #=
            uu
            @upper  no
            @lower  no
            @inlet  no
            @outlet yes
        =#
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            dpdx = (pK[i + 1, j] - pK[i, j])/dx
            uu[i, j, K] -= dt*dpdx
        end
        #=
            vv
            @upper  no
            @lower  no
            @inlet  no
            @outlet no
        =#
        for j = gc + 1:sz[2] - gc - 1, i = gc + 1:sz[1] - gc
            dpdy = (pK[i, j + 1] - pK[i, j])/dy
            vv[i, j, K] -= dt*dpdy
        end
    end
end

function div_UK!(uu, vv, divU, dx, dy, P, sz, gc)
    for K = 1:P + 1
        uuK = @view uu[:, :, K]
        vvK = @view vv[:, :, K]
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            divU[i, j, K] = cell_divUK(uuK, vvK, dx, dy, i, j)
        end
    end
    mag = norm(divU)
    inner_cell_count = (sz[1] - 2*gc)*(sz[2] - 2*gc)
    return mag/sqrt(inner_cell_count)
end