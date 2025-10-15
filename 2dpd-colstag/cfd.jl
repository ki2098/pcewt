using LinearAlgebra
using Base.Threads

include("pd.jl")

function utopia_convection(fww, fw, fc, fe, fee, uW, uc, uE, dx)
    xE = uE*(fw - 27*fc + 27*fe - fee)/(24*dx)
    xW = uW*(fww - 27*fw + 27*fc - fe)/(24*dx)
    xA = abs(uc)*(fww - 4*fw + 6*fc - 4*fe + fee)/(12*dx)
    return 0.5*(xE + xW) + xA
end

function central_difference_convection(fw, fc, fe, uW, uE, dx)
    fluxW = uW*(fc + fw)/2
    fluxE = uE*(fe + fc)/2
    return (fluxE - fluxW)/dx
end

function first_upwind_convection(fw, fc, fe, uW, uE, dx)
    fluxW = (uW*(fc + fw) - abs(uW)*(fc - fw))/2
    fluxE = (uE*(fe + fc) - abs(uE)*(fe - fc))/2
    return (fluxE - fluxW)/dx
end

function cell_convectionK(uW, uc, uE, vS, vc, vN, f, T2, T3, K, P, dx, dy, i, j)
    convection = 0.0
    
    for I = 1:P + 1

        ucI  = uc[I]
        uEI  = uE[I]
        uWI  = uW[I]
        vcI  = vc[I]
        vNI  = vN[I]
        vSI  = vS[I]

        for J = 1:P + 1
            if I == 1
                fcJ  = f[i, j, J]
                feJ  = f[i + 1, j, J]
                feeJ = f[i + 2, j, J]
                fwJ  = f[i - 1, j, J]
                fwwJ = f[i - 2, j, J]
                fnJ  = f[i, j + 1, J]
                fnnJ = f[i, j + 2, J]
                fsJ  = f[i, j - 1, J]
                fssJ = f[i, j - 2, J]
                uIdfJdx = utopia_convection(fwwJ, fwJ ,fcJ, feJ, feeJ, uWI, ucI, uEI, dx)
                vIdfJdy = utopia_convection(fssJ, fsJ, fcJ, fnJ, fnnJ, vSI, vcI, vNI, dy)
            else
                fcJ  = f[i, j, J]
                feJ  = f[i + 1, j, J]
                fwJ  = f[i - 1, j, J]
                fnJ  = f[i, j + 1, J]
                fsJ  = f[i, j - 1, J]
                uIdfJdx = central_difference_convection(fwJ, fcJ, feJ, uWI, uEI, dx)
                vIdfJdy = central_difference_convection(fsJ, fcJ, fnJ, vSI, vNI, dy)
                # uIdfJdx = first_upwind_convection(fwJ, fcJ, feJ, uWI, uEI, dx)
                # vIdfJdy = first_upwind_convection(fsJ, fcJ, fnJ, vSI, vNI, dy)
            end
            convection += (uIdfJdx + vIdfJdy)*(T3[I, J, K]/T2[K, K])
        end
    end
    return convection
end

function cell_diffusionK(fK, μ, dx, dy, i, j)
    fcK = fK[i, j]
    feK = fK[i + 1, j]
    fwK = fK[i - 1, j]
    fnK = fK[i, j + 1]
    fsK = fK[i, j - 1]
    d2fKdx2 = (feK - 2*fcK + fwK)/(dx^2)
    d2fKdy2 = (fnK - 2*fcK + fsK)/(dy^2)
    return μ*(d2fKdx2 + d2fKdy2)
end

function pseudo_U!(ut, vt, u, v, uu, vv, Umag, dfunc, T2, T3, P, μ, dx, dy, dt, sz, gc)
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            uE = uu[i    , j, :]
            uW = uu[i - 1, j, :]
            uc = ut[i    , j, :]
            vc = vt[i, j    , :]
            vS = vv[i, j - 1, :]
            vN = vv[i, j    , :]
            Umagc = Umag[i, j, :]
            dfuncc = dfunc[i, j]
            for K = 1:P + 1
                uK = @view ut[:, :, K]
                vK = @view vt[:, :, K]
                fxc, fyc = cell_PD_forceK(uc, vc, Umagc, dfuncc, T2, T3, K, P)
                uK_conv = cell_convectionK(uW, uc, uE, vS, vc, vN, ut, T2, T3, K, P, dx, dy, i, j)
                uK_diff = cell_diffusionK(uK, μ, dx, dy, i, j)
                vK_conv = cell_convectionK(uW, uc, uE, vS, vc, vN, vt, T2, T3, K, P, dx, dy, i, j)
                vK_diff = cell_diffusionK(vK, μ, dx, dy, i, j)
                u[i, j, K] = uc[K] + dt*(- uK_conv + uK_diff - fxc)
                v[i, j, K] = vc[K] + dt*(- vK_conv + vK_diff - fyc)
            end
        end
    end
end

function interpol_UU!(u, v, uu, vv, sz, gc)
    #=
        uu
        @upper  no
        @lower  no
        @inlet  no
        @outlet yes
    =#
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            uu[i, j, :] = 0.5*(u[i, j, :] + u[i + 1, j, :])
        end
    end

    #=
        vv
        @upper  no
        @lower  no
        @inlet  no
        @outlet no
    =#
    @threads for j = gc + 1:sz[2] - gc - 1
        for i = gc + 1:sz[1] - gc
            vv[i, j, :] = 0.5*(v[i, j, :] + v[i, j + 1, :])
        end
    end 
end

function pressure_eq_b!(uu, vv, b, P, dx, dy, dt, max_diag, sz, gc)
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            dudx = (uu[i, j, :] - uu[i - 1, j, :])/dx
            dvdy = (vv[i, j, :] - vv[i, j - 1, :])/dy
            b[i, j, :] = (dudx + dvdy)/(dt*max_diag)
        end
    end
end

function update_U_by_gradp!(u, v, uu, vv, p, P, dx, dy, dt, sz, gc)
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            dpdx = (p[i + 1, j, :] - p[i - 1, j, :])/(2*dx)
            dpdy = (p[i, j + 1, :] - p[i, j - 1, :])/(2*dy)
            u[i, j, :] -= dt*dpdx
            v[i, j, :] -= dt*dpdy
        end
    end

    #=
        uu
        @upper  no
        @lower  no
        @inlet  no
        @outlet yes
    =#
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            dpdx = (p[i + 1, j, :] - p[i, j, :])/dx
            uu[i, j, :] -= dt*dpdx
        end
    end

    #=
        vv
        @upper  no
        @lower  no
        @inlet  no
        @outlet no
    =#
    @threads for j = gc + 1:sz[2] - gc - 1
        for i = gc + 1:sz[1] - gc
            dpdy = (p[i, j + 1, :] - p[i, j, :])/dy
            vv[i, j, :] -= dt*dpdy
        end
    end
end

function div_UK!(uu, vv, divU, dx, dy, P, sz, gc)
    @threads for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            dudx = (uu[i, j, :] - uu[i - 1, j, :])/dx
            dvdy = (vv[i, j, :] - vv[i, j - 1, :])/dy
            divU[i, j, :] = dudx + dvdy
        end
    end
    mag = norm(divU)
    effective_cell_cnt = (sz[1] - 2*gc)*(sz[2] - 2*gc)
    return mag/sqrt(effective_cell_cnt)
end