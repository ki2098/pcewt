using CUDA

function kernel_Ubc_xminus!(u, v, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= gc && gc < j <= sz[2]-gc
        i_mirror = 2*gc + 1 - i
        for K = 1:P+1
            u[i, j, K] = - u[i_mirror, j, K]
            v[i, j, K] = - v[i_mirror, j, K]
        end
    end
    return nothing
end

function kernel_Ubc_xplus!(u, v, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= gc && gc < j <= sz[2]-gc
        i += sz[1]-gc
        i_mirror = 2*(sz[1] - gc) + 1 - i
        for K = 1:P+1
            u[i, j, K] = - u[i_mirror, j, K]
            v[i, j, K] = - v[i_mirror, j, K]
        end
        
    end
    return nothing
end

function kernel_Ubc_yminus!(u, v, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= gc
        j_mirror = 2*gc + 1 - j
        for K = 1:P+1
            u[i, j, K] = - u[i, j_mirror, K]
            v[i, j, K] = - v[i, j_mirror, K]
        end
    end
    return nothing
end

function kernel_Ubc_yplus!(u, v, ulid, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= gc
        j += sz[2]-gc
        j_mirror = 2*(sz[2] - gc) + 1 - j
        for K = 1:P+1
            u[i, j, K] = - u[i, j_mirror, K] + 2*ulid[K]
            v[i, j, K] = - v[i, j_mirror, K]
        end
    end
    return nothing
end

function gpu_Ubc!(u, v, ulid, P, sz, gc)
    nthread_x = (gc, 32)
    nblock_x = (1, cld(sz[2], nthread_x[2]))
    nthread_y = (32, gc)
    nblock_y = (cld(sz[1], nthread_y[1]), 1)
    @cuda threads=nthread_x blocks=nblock_x kernel_Ubc_xminus!(
        u, v, P, sz, gc
    )
    @cuda threads=nthread_x blocks=nblock_x kernel_Ubc_xplus!(
        u, v, P, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_Ubc_yminus!(
        u, v, P, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_Ubc_yplus!(
        u, v, ulid, P, sz, gc
    )
end

function kernel_pbc_xminus!(p, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        i = gc
        for K = 1:P+1
            p[i, j, K] = p[i + 1, j, K]
        end
    end
    return nothing
end

function kernel_pbc_xplus!(p, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 1 && gc < j <= sz[2]-gc
        i = sz[1] - gc + 1
        for K = 1:P+1
            p[i, j, K] = p[i - 1, j, K]
        end
    end
    return nothing
end

function kernel_pbc_yminus!(p, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        j = gc
        for K = 1:P+1
            p[i, j, K] = p[i, j + 1, K]
        end
    end
    return nothing
end

function kernel_pbc_yplus!(p, P, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && j <= 1
        j = sz[2] - gc + 1
        for K = 1:P+1
            p[i, j, K] = p[i, j - 1, K]
        end
    end
    return nothing
end

function gpu_pbc!(p, P, sz, gc)
    nthread_x = (1, 32)
    nblock_x = (1, cld(sz[2], nthread_x[2]))
    nthread_y = (32, 1)
    nblock_y = (cld(sz[1], nthread_y[1]), 1)
    @cuda threads=nthread_x blocks=nblock_x kernel_pbc_xminus!(
        p, P, sz, gc
    )
    @cuda threads=nthread_x blocks=nblock_x kernel_pbc_xplus!(
        p, P, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_pbc_yminus!(
        p, P, sz, gc
    )
    @cuda threads=nthread_y blocks=nblock_y kernel_pbc_yplus!(
        p, P, sz, gc
    )
end

# checked