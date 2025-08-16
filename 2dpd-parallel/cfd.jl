include("pdmodel.jl")

function utopia_convection(fww, fw, fc, fe, fee, u, dx)
    return (u*(- fee + 8*fe - 8*fw + fww) + abs(u)*(fee - 4*fe + 6*fc - 4*fw + fww))/(12*dx)
end

function first_order_upwind(fw, fc, fe, u, dx)
    return (u*(fe - fw) + abs(u)*(- fe + 2*fc - fw))/(2*dx)
end

function central_difference_convection(fw, fe, u, dx)
    return u*(fe - fw)/(2*dx)
end

function cell_convectionK(uc, vc, f, T2, T3, K, P, dx, dy, i, j)
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
        if I == 1
            convection_fJx = utopia_convection(fwwJ, fwJ, fcJ, feJ, feeJ, ucI, dx) 
            convection_fJy = utopia_convection(fssJ, fsJ, fcJ, fnJ, fnnJ, vcI, dy)
        else
            convection_fJx = central_difference_convection(fwJ, feJ, ucI, dx)
            convection_fJy = central_difference_convection(fsJ, fnJ, vcI, dy)
        end
        convection += (convection_fJx + convection_fJy)*(T3[I, J, K]/T2[K, K])
    end
    return convection
end

function cell_diffusionK(fK, viscosity, dx, dy, i, j)
    fcK = fK[i, j]
    feK = fK[i + 1, j]
    fwK = fK[i - 1, j]
    fnK = fK[i, j + 1]
    fsK = fK[i, j - 1]
    return viscosity*((feK - 2*fcK + fwK)/(dx^2) + (fnK - 2*fcK + fsK)/(dy^2))
end

function pseudo_U!(uold, vold, u, v, Umag, dfunc, T2, T3, P, viscosity, dx, dy, dt, sz, gc)
    for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            uc = uold[i, j, :]
            vc = vold[i, j, :]
            Umagc = Umag[i, j, :]
            dfuncc = dfunc[i, j]
            for K = 1:P + 1
                fxc, fyc = cell_PD_forceK(uc, vc, Umagc, dfuncc, T2, T3, K, P)
                uK_convection = cell_convectionK(uc, vc, uold, T2, T3, K, P, dx, dy, i, j)
                uK_diffusion = cell_diffusionK((@view uold[:, :, K]), viscosity, dx, dy, i, j)
                vK_convection = cell_convectionK(uc, vc, vold, T2, T3, K, P, dx, dy, i, j)
                vK_diffusion = cell_diffusionK((@view vold[:, :, K]), viscosity, dx, dy, i, j)
                u[i, j, K] = uc[K] + dt*(- uK_convection + uK_diffusion - fxc)
                v[i, j, K] = vc[K] + dt*(- vK_convection + vK_diffusion - fyc)
            end
        end
    end
end

function cell_div_UK(uK, vK, dx, dy, i, j)
    ueK = uK[i + 1, j]
    uwK = uK[i - 1, j]
    vnK = vK[i, j + 1]
    vsK = vK[i, j - 1]
    return (ueK - uwK)/(2*dx) + (vnK - vsK)/(2*dy)
end

function pressure_eq_b!(u, v, b, P, dx, dy, dt, max_diagA, sz, gc)
    for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            for K = 1:P + 1
                uK = @view u[:, :, K]
                vK = @view v[:, :, K]
                b[i, j, K] = cell_div_UK(uK, vK, dx, dy, i, j)/(dt*max_diagA)
            end
        end
    end
end

function cell_grad_pK(pK, dx, dy, i, j)
    peK = pK[i + 1, j]
    pwK = pK[i - 1, j]
    pnK = pK[i, j + 1]
    psK = pK[i, j - 1]
    return (peK - pwK)/(2*dx), (pnK - psK)/(2*dy)
end

function update_U_by_grad_p!(u, v, p, P, dx, dy, dt, sz, gc)
    for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            for K = 1:P + 1
                pK = @view p[:, :, K]
                dpKdx, dpKdy = cell_grad_pK(pK, dx, dy, i, j)
                u[i, j, K] -= dt*dpKdx
                v[i, j, K] -= dt*dpKdy
            end
        end
    end
end

function div_UK!(u, v, divU, dx, dy, P, sz, gc)
    for j = gc + 1:sz[2] - gc
        for i = gc + 1:sz[1] - gc
            for K = 1:P + 1
                uK = @view u[:, :, K]
                vK = @view v[:, :, K]
                divU[i, j, K] = cell_div_UK(uK, vK, dx, dy, i, j)
            end
        end
    end
    mag = norm(divU)
    inner_cell_count = (sz[1] - 2*gc)*(sz[2] - 2*gc)
    return mag/sqrt(inner_cell_count)
end