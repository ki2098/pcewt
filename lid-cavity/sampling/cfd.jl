using LinearAlgebra
using CUDA

function utopia_convection(fww, fw, fc, fe, fee, u, dx)
    return (u*(- fee + 8*fe - 8*fw + fww) + abs(u)*(fee - 4*fe + 6*fc - 4*fw + fww))/(12*dx)
end

function kernel_pseudo_U!(unew, vnew, u, v, μ, dx, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        uc  = u[i    , j]
        ue  = u[i + 1, j]
        uee = u[i + 2, j]
        uw  = u[i - 1, j]
        uww = u[i - 2, j]
        un  = u[i, j + 1]
        unn = u[i, j + 2]
        us  = u[i, j - 1]
        uss = u[i, j - 2]
        vc  = v[i    , j]
        ve  = v[i + 1, j]
        vee = v[i + 2, j]
        vw  = v[i - 1, j]
        vww = v[i - 2, j]
        vn  = v[i, j + 1]
        vnn = v[i, j + 2]
        vs  = v[i, j - 1]
        vss = v[i, j - 2]

        uux = utopia_convection(
            uww, uw, uc, ue, uee, uc, dx
        )
        vuy = utopia_convection(
            uss, us, uc, un, unn, vc, dx
        )
        uvx = utopia_convection(
            vww, vw, vc, ve, vee, uc, dx
        )
        vvy = utopia_convection(
            vss, vs, vc, vn, vnn, vc, dx
        )
        uxx = (ue - 2*uc + uw)/(dx^2)
        uyy = (un - 2*uc + us)/(dx^2)
        vxx = (ve - 2*vc + vw)/(dx^2)
        vyy = (vn - 2*vc + vs)/(dx^2)
        unew[i, j] = uc + dt*(- (uux + vuy) + μ*(uxx + uyy))
        vnew[i, j] = vc + dt*(- (uvx + vvy) + μ*(vxx + vyy))
    end
    nothing
end

function cell_divU(u, v, dx, i, j)
    ue = u[i + 1, j]
    uw = u[i - 1, j]
    vn = v[i, j + 1]
    vs = v[i, j - 1]
    return (ue - uw + vn - vs)/(2*dx)
end

function kernel_pressure_eq_b!(u, v, b, dx, dt, maxdiag, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        b[i, j] = cell_divU(u, v, dx, i, j)/(dt*maxdiag)
    end
    nothing
end

function gpu_pseudo_U!(unew, vnew, u, v, b, μ, dx, dt, maxdiag, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_pseudo_U!(
        unew, vnew, u, v, μ, dx, dt, sz, gc
    )
    @cuda threads=nthread blocks=nblock kernel_pressure_eq_b!(
        unew, vnew, b, dx, dt, maxdiag, sz, gc
    )
end

function kernel_update_U!(u, v, p, dx, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        u[i, j] -= dt*(p[i+1, j] - p[i-1, j])/(2*dx)
        v[i, j] -= dt*(p[i, j+1] - p[i, j-1])/(2*dx)
    end
    nothing
end

function gpu_update_U!(u, v, p, dx, dt, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_update_U!(
        u, v, p, dx, dt, sz, gc
    )
end

function kernel_divU!(u, v, divU, dx, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        divU[i, j] = cell_divU(u, v, dx, i, j)
    end
    nothing
end

function gpu_divU!(u, v, divU, dx, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_divU!(
        u, v, divU, dx, sz, gc
    )
    mag = norm(divU)
    ncell = prod(sz .- 2*gc)
    mag = mag/sqrt(ncell)
    return mag
end