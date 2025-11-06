using LinearAlgebra
using CUDA

function utopia_convection(fww, fw, fc, fe, fee, u, dx)
    return (u*(- fee + 8*fe - 8*fw + fww) + abs(u)*(fee - 4*fe + 6*fc - 4*fw + fww))/(12*dx)
end

function cell_convectionK(u, v, f, T2, T3, K, P, dx, i, j)
    convection = 0.0
    for I = 1:P+1, J = 1:P+1
        m = T3[I, J, K]/T2[K, K]
        ucI  = u[i, j, I]
        vcI  = v[i, j, I]
        fcJ  = f[i, j, J]
        feJ  = f[i + 1, j, J]
        feeJ = f[i + 2, j, J]
        fwJ  = f[i - 1, j, J]
        fwwJ = f[i - 2, j, J]
        fnJ  = f[i, j + 1, J]
        fnnJ = f[i, j + 2, J]
        fsJ  = f[i, j - 1, J]
        fssJ = f[i, j - 2, J]
        convection += (
            utopia_convection(fwwJ, fwJ, fcJ, feJ, feeJ, ucI, dx) 
        +   utopia_convection(fssJ, fsJ, fcJ, fnJ, fnnJ, vcI, dx)
        )*m
    end
    return convection
end

function cell_diffusionK(f, viscosity, K, dx, i, j)
    fcK = f[i, j, K]
    feK = f[i + 1, j, K]
    fwK = f[i - 1, j, K]
    fnK = f[i, j + 1, K]
    fsK = f[i, j - 1, K]
    return viscosity*(feK + fwK + fnK + fsK - 4*fcK)/(dx^2)
end

function kernel_pseudo_U!(uold, vold, u, v, T2, T3, P, viscosity, dx, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        for K = 1:P+1
            uK_convection = cell_convectionK(uold, vold, uold, T2, T3, K, P, dx, i, j)
            vK_convection = cell_convectionK(uold, vold, vold, T2, T3, K, P, dx, i, j)
            uK_diffusion = cell_diffusionK(uold, viscosity, K, dx, i, j)
            vK_diffusion = cell_diffusionK(vold, viscosity, K, dx, i, j)
            u[i, j, K] = uold[i, j, K] + dt*(- uK_convection + uK_diffusion)
            v[i, j, K] = vold[i, j, K] + dt*(- vK_convection + vK_diffusion)
        end
    end
    return nothing
end

function cell_divUK(u, v, K, dx, i, j)
    ueK = u[i + 1, j, K]
    uwK = u[i - 1, j, K]
    vnK = v[i, j + 1, K]
    vsK = v[i, j - 1, K]
    return (ueK - uwK + vnK - vsK)/(2*dx)
end

function kernel_pressure_eq_b!(u, v, b, P, dx, dt, scale, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        for K = 1:P+1
            b[i, j, K] = cell_divUK(u, v, K, dx, i, j)/(dt*scale)
        end
    end
    return nothing
end

function gpu_pseudo_U!(uold, vold, u, v, b, T2, T3, P, viscosity, dx, dt, maxdiag, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_pseudo_U!(
        uold, vold, u, v, T2, T3, P, viscosity, dx, dt, sz, gc
    )
    @cuda threads=nthread blocks=nblock kernel_pressure_eq_b!(
        u, v, b, P, dx, dt, maxdiag, sz, gc
    )
end

function cell_gradpK(p, K, dx, i, j)
    peK = p[i + 1, j, K]
    pwK = p[i - 1, j, K]
    pnK = p[i, j + 1, K]
    psK = p[i, j - 1, K]
    return (peK - pwK)/(2*dx), (pnK - psK)/(2*dx)
end

function kernel_update_U!(u, v, p, P, dx, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        for K = 1:P+1
            dpKdx, dpKdy = cell_gradpK(p, K, dx, i, j)
            u[i, j, K] -= dt*dpKdx
            v[i, j, K] -= dt*dpKdy
        end
    end
    return nothing
end

function gpu_update_U!(u, v, p, P, dx, dt, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_update_U!(
        u, v, p, P, dx, dt, sz, gc
    )
end

function kernel_divU!(u, v, divU, P, dx, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        for K = 1:P+1
            divU[i, j, K] = cell_divUK(u, v, K, dx, i, j)
        end
    end
    return nothing
end

function gpu_divU!(u, v, divU, P, dx, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_divU!(
        u, v, divU, P, dx, sz, gc
    )
    mag = norm(divU)
    ncell = prod(sz .- 2*gc)
    mag = mag / sqrt(ncell)
    return mag
end

# checked