function utopia_convection(fww, fw, fc, fe, fee, u, dx)
    return (u*(- fee + 8*fe - 8*fw + fww) + abs(u)*(fee - 4*fe + 6*fc - 4*fw + fww))/(12*dx)
end

function cell_convectionK(uc, vc, f, M, K, P, dx, i, j)
    convection = 0.0
    for I = 1:P + 1, J = 1:P + 1
        MIJK = M[I, J, K]
        ucI = uc[I]
        vcI = vc[I]
        fJ = @view f[:, :, J]
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
        )*MIJK
    end
    return convection
end

function cell_diffusionK(fK, viscosity, dx, i, j)
    fcK = fK[i, j]
    feK = fK[i + 1, j]
    fwK = fK[i - 1, j]
    fnK = fK[i, j + 1]
    fsK = fK[i, j - 1]
    return viscosity*(feK + fwK + fnK + fsK - 4*fcK)/(dx^2)
end

function pseudo_U!(uold, vold, u, v, M, P, viscosity, dx, dt, sz, gc)
    for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
        uc = uold[i, j, :]
        vc = vold[i, j, :]
        for K = 1:P + 1
            uK_convection = cell_convectionK(uc, vc, uold, M, K, P, dx, i, j)
            uK_diffusion = cell_diffusionK((@view uold[:, :, K]), viscosity, dx, i, j)
            vK_convection = cell_convectionK(uc, vc, vold, M, K, P, dx, i, j)
            vK_diffusion = cell_diffusionK((@view vold[:, :, K]), viscosity, dx, i, j)
            u[i, j, K] = uc[K] + dt*(- uK_convection + uK_diffusion)
            v[i, j, K] = vc[K] + dt*(- vK_convection + vK_diffusion)
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

function pressure_eq_b!(u, v, b, P, dx, dt, scale, sz, gc)
    for K = 1:P + 1
        uK = @view u[:, :, K]
        vK = @view v[:, :, K]
        for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
            b[i, j, K] = cell_div_UK(uK, vK, dx, i, j)/(dt*scale)
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
        uK = @view u[:, :, K]
        vK = @view v[:, :, K]
        pK = @view p[:, :, K]
        for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
            dpKdx, dpKdy = cell_grad_pK(pK, dx, i, j)
            uK[i, j] -= dt*dpKdx
            vK[i, j] -= dt*dpKdy
        end
    end
end

function div_UK!(u, v, divU, dx, sz, gc)
    for K = 1:P + 1
        uK = @view u[:, :, K]
        vK = @view v[:, :, K]
        for i = gc + 1:sz[1] - gc, j = gc + 1:sz[2] - gc
            divU[i, j, K] = cell_div_UK(uK, vK, dx, i, j)
        end
    end
    mag = norm(divU)
    inner_cell_count = (sz[1] - 2*gc)*(sz[2] - 2*gc)
    return mag/sqrt(inner_cell_count)
end