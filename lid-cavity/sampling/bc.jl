using CUDA

function kernel_Ubc_xminus!(u, v, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= gc && gc < j <= sz[2]-gc
        i_mirror = 2*gc + 1 - i
        u[i, j] = - u[i_mirror, j]
        v[i, j] = - v[i_mirror, j]
    end
    return nothing
end

function kernel_Ubc_xplus!(u, v, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= gc && gc < j <= sz[2]-gc
        i += sz[1]-gc
        i_mirror = 2*(sz[1] - gc) + 1 - i
        u[i, j] = - u[i_mirror, j]
        v[i, j] = - v[i_mirror, j]
    end
    return nothing
end

function kernel_Ubc_yminus!(u, v, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= gc
        j_mirror = 2*gc + 1 - j
        u[i, j] = - u[i, j_mirror]
        v[i, j] = - v[i, j_mirror]
    end
    return nothing
end

function kernel_Ubc_yplus!(u, v, ulid, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= gc
        j += sz[2]-gc
        j_mirror = 2*(sz[2] - gc) + 1 - j
        u[i, j] = 2*ulid - u[i, j_mirror]
        v[i, j] = - v[i, j_mirror]
    end
    return nothing
end

function gpu_Ubc!(u, v, ulid, sz, gc)
    nthread_x = (gc, 32)
    nblock_x = (1, cld(sz[2], nthread_x[2]))
    nthread_y = (32, gc)
    nblock_y = (cld(sz[1], nthread_y[1]), 1)
    @cuda threads=nthread_x blocks=nblock_x kernel_Ubc_xminus!(
        u, v, sz, gc
    )
    @cuda threads=nthread_x blocks=nblock_x kernel_Ubc_xplus!(
        u, v, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_Ubc_yminus!(
        u, v, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_Ubc_yplus!(
        u, v, ulid, sz, gc
    )
end

function kernel_uubc_xminus!(uu, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        i = gc
        uu[i, j] = 0
    end
    return nothing
end

function kernel_uubc_xplus!(uu, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        i = sz[1] - gc
        uu[i, j] = 0
    end
    return nothing
end

function kernel_vvbc_yminus!(vv, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        j = gc
        vv[i, j] = 0
    end
    return nothing
end

function kernel_vvbc_yplus!(vv, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        j = sz[2] - gc
        vv[i, j] = 0
    end
    return nothing
end

function gpu_UUbc!(uu, vv, sz, gc)
    nthread_x = (1, 32)
    nblock_x = (1, cld(sz[2], nthread_x[2]))
    nthread_y = (32, 1)
    nblock_y = (cld(sz[1], nthread_y[1]), 1)
    @cuda threads=nthread_x blocks=nblock_x kernel_uubc_xminus!(
        uu, sz, gc
    )
    @cuda threads=nthread_x blocks=nblock_x kernel_uubc_xplus!(
        uu, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_vvbc_yminus!(
        vv, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_vvbc_yplus!(
        vv, sz, gc
    )
end

function kernel_pbc_xminus!(p, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        i = gc
        p[i, j] = p[i + 1, j]
    end
    return nothing
end

function kernel_pbc_xplus!(p, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        i = sz[1] - gc + 1
        p[i, j] = p[i - 1, j]
    end
    return nothing
end

function kernel_pbc_yminus!(p, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        j = gc
        p[i, j] = p[i, j + 1]
    end
    return nothing
end

function kernel_pbc_yplus!(p, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        j = sz[2] - gc + 1
        p[i, j] = p[i, j - 1]
    end
    return nothing
end

function gpu_pbc!(p, sz, gc)
    nthread_x = (1, 32)
    nblock_x = (1, cld(sz[2], nthread_x[2]))
    nthread_y = (32, 1)
    nblock_y = (cld(sz[1], nthread_y[1]), 1)
    @cuda threads=nthread_x blocks=nblock_x kernel_pbc_xminus!(
        p, sz, gc
    )
    @cuda threads=nthread_x blocks=nblock_x kernel_pbc_xplus!(
        p, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_pbc_yminus!(
        p, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_pbc_yplus!(
        p, sz, gc
    )
end