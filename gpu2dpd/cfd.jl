using CUDA

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

function cell_diffusionK(f, μ, K, dx, dy, i, j)
    fcK = f[i, j, K]
    feK = f[i + 1, j, K]
    fwK = f[i - 1, j, K]
    fnK = f[i, j + 1, K]
    fsK = f[i, j - 1, K]
    d2fKdx2 = (feK - 2*fcK + fwK)/(dx^2)
    d2fKdy2 = (fnK - 2*fcK + fsK)/(dy^2)
    return μ*(d2fKdx2 + d2fKdy2)
end

function kernel_pseudo_U!(ut, vt, u, v, uu, vv, Umag, dfunc, T2, T3, P, μ, dx, dy, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if (
        gc+1 <= i <= sz[1]-gc && 
        gc+1 <= j <= sz[2]-gc
    )
        uE = uu[i    , j, :]
        uW = uu[i - 1, j, :]
        uc = ut[i    , j, :]
        vc = vt[i, j    , :]
        vS = vv[i, j - 1, :]
        vN = vv[i, j    , :]
        Umagc = Umag[i, j, :]
        dfuncc = dfunc[i, j]
        for K = 1:P + 1
            fxc, fyc = 0, 0
            uK_conv = cell_convectionK(uW, uc, uE, vS, vc, vN, ut, T2, T3, K, P, dx, dy, i, j)
            uK_diff = cell_diffusionK(ut, μ, K, dx, dy, i, j)
            vK_conv = cell_convectionK(uW, uc, uE, vS, vc, vN, vt, T2, T3, K, P, dx, dy, i, j)
            vK_diff = cell_diffusionK(vt, μ, K, dx, dy, i, j)
            u[i, j, K] = uc[K] + dt*(- uK_conv + uK_diff - fxc)
            v[i, j, K] = vc[K] + dt*(- vK_conv + vK_diff - fyc)
        end
    end
    return nothing
end

function kernel_interpolate_U!(u, v, uu, vv, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y

    if (
        gc+1 <= i <= sz[1]-gc && 
        gc+1 <= j <= sz[2]-gc
    )
        uu[i, j, :] = 0.5*(u[i, j, :] + u[i + 1, j, :])
    end

    if (
        gc+1 <= i <= sz[1]-gc && 
        gc+1 <= j <= sz[2]-gc-1
    )
        vv[i, j, :] = 0.5*(v[i, j, :] + v[i, j + 1, :])
    end
    return nothing
end

function kernel_pressure_eq_b!(uu, vv, b, dx, dy, dt, max_diag, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y

    if (
        gc+1 <= i <= sz[1]-gc && 
        gc+1 <= j <= sz[2]-gc
    )
        dudx = (uu[i, j, :] - uu[i - 1, j, :])/dx
        dvdy = (vv[i, j, :] - vv[i, j - 1, :])/dy
        b[i, j, :] = (dudx + dvdy)/(dt*max_diag)
    end
end

function kernel_update_U!(u, v, uu, vv, p, dx, dy, dt, sz, gc)
    
end