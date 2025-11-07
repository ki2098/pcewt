using LinearAlgebra
using CUDA

function utopia_convection(fww, fw, fc, fe, fee, uW, uc, uE, dx)
    xE = uE*(fw - 27*fc + 27*fe - fee)/(24*dx)
    xW = uW*(fww - 27*fw + 27*fc - fe)/(24*dx)
    xA = abs(uc)*(fww - 4*fw + 6*fc - 4*fe + fee)/(12*dx)
    return 0.5*(xE + xW) + xA
end

function cell_convection(f, uW, uc, uE, vS, vc, vN, dx, dy, i, j)
    fc  = f[i, j]
    fe  = f[i + 1, j]
    fee = f[i + 2, j]
    fw  = f[i - 1, j]
    fww = f[i - 2, j]
    fn  = f[i, j + 1]
    fnn = f[i, j + 2]
    fs  = f[i, j - 1]
    fss = f[i, j - 2]
    udfdx = utopia_convection(fww, fw, fc, fe, fee, uW, uc, uE, dx)
    vdfdy = utopia_convection(fss, fs, fc, fn, fnn, vS, vc, vN, dy)
    return udfdx + vdfdy
end

function cell_diffusion(f, μ, dx, dy, i, j)
    fc = f[i, j]
    fe = f[i + 1, j]
    fw = f[i - 1, j]
    fn = f[i, j + 1]
    fs = f[i, j - 1]
    d2fdx2 = (fe - 2*fc + fw)/(dx^2)
    d2fdy2 = (fn - 2*fc + fs)/(dy^2)
    return μ*(d2fdx2 + d2fdy2)
end

function kernel_pseudo_U!(ut, vt, u, v, uu, vv, dx, dy, dt, μ, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        uc = ut[i, j]
        vc = vt[i, j]
        uE = uu[i, j]
        uW = uu[i - 1, j]
        vN = vv[i, j]
        vS = vv[i, j - 1]
        u_conv = cell_convection(ut, uW, uc, uE, vS, vc, vN, dx, dy, i, j)
        u_diff = cell_diffusion(ut, μ, dx, dy, i, j)
        v_conv = cell_convection(vt, uW, uc, uE, vS, vc, vN, dx, dy, i, j)
        v_diff = cell_diffusion(vt, μ, dx, dy, i, j)
        u[i, j] = uc + dt*(- u_conv + u_diff)
        v[i, j] = vc + dt*(- v_conv + v_diff)
    end
    return nothing
end

function kernel_interpolate_uu!(u, uu, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc-1 && gc < j <= sz[2]-gc
        uu[i, j] = (u[i, j] + u[i + 1, j])/2
    end
    return nothing
end

function kernel_interpolate_vv!(v, vv, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc-1
        vv[i, j] = (v[i, j] + v[i, j + 1])/2
    end
    return nothing
end

function cell_div_U(uu, vv, dx, dy, i, j)
    uE = uu[i, j]
    uW = uu[i - 1, j]
    vN = vv[i, j]
    vS = vv[i, j - 1]
    return (uE - uW)/dx + (vN - vS)/dy
end

function kernel_pressure_eq_rhs!(uu, vv, b, dx, dy, dt, max_diag, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        b[i, j] = cell_div_U(uu, vv, dx, dy, i, j)/(dt*max_diag)
    end
    return nothing
end

function kernel_update_U_by_grad_p!(u, v, p, dx, dy, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        dpdx = (p[i + 1, j] - p[i - 1, j])/(2*dx)
        dpdy = (p[i, j + 1] - p[i, j - 1])/(2*dy)
        u[i, j] -= dt*dpdx
        v[i, j] -= dt*dpdy
    end
    return nothing
end

function kernel_update_uu_by_dpdx!(uu, p, dx, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc-1 && gc < j <= sz[2]-gc
        dpdx = (p[i + 1, j] - p[i, j])/dx
        uu[i, j] -= dt*dpdx
    end
    return nothing
end

function kernel_update_vv_by_dpdy!(vv, p, dy, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc-1
        dpdy = (p[i, j + 1] - p[i, j])/dy
        vv[i, j] -= dt*dpdy
    end
    return nothing
end

function kernel_div_U!(uu, vv, div_U, dx, dy, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        div_U[i, j] = cell_div_U(uu, vv, dx, dy, i, j)
    end
    return nothing
end

function gpu_predict_U!(ut, vt, u, v, uu, vv, dx, dy, dt, μ, sz, gc, nthread)
    nblock = (
        cld(sz[1], nthread[1]),
        cld(sz[2], nthread[2])
    )
    @cuda threads=nthread blocks=nblock kernel_pseudo_U!(
        ut, vt, u, v, uu, vv,
        dx, dy, dt, μ,
        sz, gc
    )
    @cuda threads=nthread blocks=nblock kernel_interpolate_uu!(
        u, uu,
        sz, gc
    )
    @cuda threads=nthread blocks=nblock kernel_interpolate_vv!(
        v, vv,
        sz, gc
    )
end

function gpu_pressure_eq_b!(uu, vv, b, dx, dy, dt, max_diag, sz, gc, nthread)
    nblock = (
        cld(sz[1], nthread[1]),
        cld(sz[2], nthread[2])
    )
    @cuda threads=nthread blocks=nblock kernel_pressure_eq_rhs!(
        uu, vv, b,
        dx, dy, dt, max_diag,
        sz, gc
    ) 
end

function gpu_update_U!(u, v, uu, vv, p, dx, dy, dt, sz, gc, nthread)
    nblock = (
        cld(sz[1], nthread[1]),
        cld(sz[2], nthread[2])
    )
    @cuda threads=nthread blocks=nblock kernel_update_U_by_grad_p!(
        u, v, p,
        dx, dy, dt,
        sz, gc
    )
    @cuda threads=nthread blocks=nblock kernel_update_uu_by_dpdx!(
        uu, p,
        dx, dt,
        sz, gc
    )
    @cuda threads=nthread blocks=nblock kernel_update_vv_by_dpdy!(
        vv, p,
        dy, dt,
        sz, gc
    )
end

function gpu_div_U!(uu, vv, div_U, dx, dy, sz, gc, nthread)
    nblock = (
        cld(sz[1], nthread[1]),
        cld(sz[2], nthread[2])
    )
    @cuda threads=nthread blocks=nblock kernel_div_U!(
        uu, vv, div_U,
        dx, dy,
        sz, gc
    )
    err = norm(div_U)
    ncell = prod(sz .- 2*gc)
    err = err/sqrt(ncell)
    return err
end