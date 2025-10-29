using CUDA

function left_side_dfdx(fww, fw, fc, dx)
    return (3*fc - 4*fw + fww)/(2*dx)
end

function kernel_Ubc_x!(u, v, ut, vt, uin, dx, dt, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= gc && gc < j <= sz[2]-gc
        u[i, j] = uin
        v[i, j] = 0

        i = i+sz[1]-gc
        uc  = ut[i  , j]
        uw  = ut[i-1, j]
        uww = ut[i-2, j]
        vc  = vt[i  , j]
        vw  = vt[i-1, j]
        vww = vt[i-2, j]
        ux = left_side_dfdx(uww, uw, uc, dx)
        vx = left_side_dfdx(vww, vw, vc, dx)
        u[i, j] = uc - dt*uc*ux
        v[i, j] = vc - dt*vc*vx
    end
    nothing
end

function kernel_Ubc_y!(u, v, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= gc
        j_mirror = 2*gc+1-j
        u[i, j] =  u[i, j_mirror]
        v[i, j] = -v[i, j_mirror]

        j = j+sz[2]-gc
        j_mirror = 2*(sz[2]-gc)+1-j
        u[i, j] =  u[i, j_mirror]
        v[i, j] = -v[i, j_mirror]
    end
    nothing
end

function kernel_uubc_xminus!(uu, uin, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        uu[gc, j] = uin
    end
    nothing
end

function kernel_vvbc!(vv, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        vv[i,       gc] = 0
        vv[i, sz[2]-gc] = 0
    end
    nothing
end

function kernel_pbc_x!(p, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        p[gc, j] = p[gc+1, j]
        p[sz[1]-gc+1, j] = p[sz[1]-gc, j]
    end
    nothing
end

function kernel_pbc_y!(p, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        p[i, gc] = p[i, gc+1]
        p[i, sz[2]-gc+1] = p[i, sz[2]-gc]
    end
    nothing
end

function gpu_Ubc!(u, v, ut, vt, uin, dx, dt, sz, gc)
    nthread = (gc, 32)
    nblock = (1, cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_Ubc_x!(
        u, v, ut, vt, uin, dx, dt, sz, gc
    )

    nthread = (32, gc)
    nblock = (cld(sz[1], nthread[1]), 1)
    @cuda threads=nthread blocks=nblock kernel_Ubc_y!(
        u, v, sz, gc
    )
end

function gpu_UUbc!(uu, vv, uin, sz, gc)
    nthread = (1, 32)
    nblock = (1, cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_uubc_xminus!(
        uu, uin, sz, gc
    )

    nthread = (32, 1)
    nblock = (cld(sz[1], nthread[1]), 1)
    @cuda threads=nthread blocks=nblock kernel_vvbc!(
        vv, sz, gc
    )
end

function gpu_pbc!(p, sz, gc)
    nthread = (1, 32)
    nblock = (1, cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_pbc_x!(
        p, sz, gc
    )

    nthread = (32, 1)
    nblock = (cld(sz[1], nthread[1]), 1)
    @cuda threads=nthread blocks=nblock kernel_pbc_y!(
        p, sz, gc
    )
end