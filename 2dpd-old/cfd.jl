include("pdmodel.jl")

function utopia_convection(fww, fw, fc, fe, fee, u, dx)
    return (u*(- fee + 8*fe - 8*fw + fww) + abs(u)*(fee - 4*fe + 6*fc - 4*fw + fww))/(12*dx)
end

function cell_convectionK(uc, vc, f, T2, T3, K, P, dx, i, j)
    convection = 0.0
    for J = 1:P + 1, I = 1:P + 1
        fJ = @view f[:, :, J]
        ucI  = uc[I]
        vcI  = vc[I]
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
            utopia_convection(fwwJ, fwJ, fcJ, feJ, feeJ, ucI, dx) 
        +   utopia_convection(fssJ, fsJ, fcJ, fnJ, fnnJ, vcI, dx)
        )*T3[I, J, K]
    end
    return convection/T2[K, K]
end

function cell_diffusionK(fK, viscosity, dx, i, j)
    fcK = fK[i, j]
    feK = fK[i + 1, j]
    fwK = fK[i - 1, j]
    fnK = fK[i, j + 1]
    fsK = fK[i, j - 1]
    return viscosity*(feK + fwK + fnK + fsK - 4*fcK)/(dx^2)
end

function pseudo_U!(uold, vold, u, v, Umag, dfunc, T2, T3, P, viscosity, dx, dt, sz, gc)
    for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
        uc = uold[i, j, :]
        vc = vold[i, j, :]
        Umagc = Umag[i, j, :]
        dfuncc = dfunc[i, j]
        for K = 1:P + 1
            fxc, fyc = cell_PD_forceK(uc, vc, Umagc, dfuncc, T2, T3, K, P)
            uK_convection = cell_convectionK(uc, vc, uold, T2, T3, K, P, dx, i, j)
            uK_diffusion = cell_diffusionK((@view uold[:, :, K]), viscosity, dx, i, j)
            vK_convection = cell_convectionK(uc, vc, vold, T2, T3, K, P, dx, i, j)
            vK_diffusion = cell_diffusionK((@view vold[:, :, K]), viscosity, dx, i, j)
            u[i, j, K] = uc[K] + dt*(- uK_convection + uK_diffusion - fxc)
            v[i, j, K] = vc[K] + dt*(- vK_convection + vK_diffusion - fyc)
        end
    end
end

function cell_div_UK(uK, vK, dx, i, j)
    ueK = uK[i + 1, j]
    uwK = uK[i - 1, j]
    vnK = vK[i, j + 1]
    vsK = vK[i, j - 1]
    return (ueK - uwK + vnK - vsK)/(2*dx)
end

function pressure_eq_b!(u, v, b, P, dx, dt, max_diagA, sz, gc)
    for K = 1:P + 1
        uK = @view u[:, :, K]
        vK = @view v[:, :, K]
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            b[i, j, K] = cell_div_UK(uK, vK, dx, i, j)/(dt*max_diagA)
        end
    end
end

function cell_grad_pK(pK, dx, i, j)
    peK = pK[i + 1, j]
    pwK = pK[i - 1, j]
    pnK = pK[i, j + 1]
    psK = pK[i, j - 1]
    return (peK - pwK)/(2*dx), (pnK - psK)/(2*dx)
end

function update_U_by_grad_p!(u, v, p, P, dx, dt, sz, gc)
    for K = 1:P + 1
        pK = @view p[:, :, K]
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            dpKdx, dpKdy = cell_grad_pK(pK, dx, i, j)
            u[i, j, K] -= dt*dpKdx
            v[i, j, K] -= dt*dpKdy
        end
    end
end

function div_UK!(u, v, divU, dx, P, sz, gc)
    for K = 1:P + 1
        uK = @view u[:, :, K]
        vK = @view v[:, :, K]
        for j = gc + 1:sz[2] - gc, i = gc + 1:sz[1] - gc
            divU[i, j, K] = cell_div_UK(uK, vK, dx, i, j)
        end
    end
    mag = norm(divU)
    inner_cell_count = (sz[1] - 2*gc)*(sz[2] - 2*gc)
    return mag/sqrt(inner_cell_count)
end