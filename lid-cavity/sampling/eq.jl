using LinearAlgebra
using CUDA

function kernel_pressure_eq_A!(A, dx, dy, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        Ae = Aw = 1/(dx^2)
        An = As = 1/(dy^2)
        Ac = - (Ae + Aw + An + As)
        A[i, j, 1] = Ac
        A[i, j, 2] = Ae
        A[i, j, 3] = Aw
        A[i, j, 4] = An
        A[i, j, 5] = As
    end
    return nothing
end

function gpu_pressure_eq_A(dx, dy, sz, gc, nthread)
    nblock = (
        cld(sz[1], nthread[1]),
        cld(sz[2], nthread[2])
    )
    A = CUDA.zeros(sz..., 5)
    A[:, :, 1] .= 1
    @cuda threads=nthread blocks=nblock kernel_pressure_eq_A!(
        A,
        dx, dy,
        sz, gc
    )
    max_diag = maximum(abs.(A[:, :, 1]))
    A ./= max_diag
    return A, max_diag
end

function gpu_init_pressure_eq(dx, dy, sz, gc, nthread)
    A, max_diag = gpu_pressure_eq_A(dx, dy, sz, gc, nthread)
    b = CUDA.zeros(sz...)
    r = CUDA.zeros(sz...)
    return A, b, r, max_diag
end

function cell_residual(A, x, b, i, j)
    xc = x[i, j]
    xe = x[i + 1, j]
    xw = x[i - 1, j]
    xn = x[i, j + 1]
    xs = x[i, j - 1]
    Ac = A[i, j, 1]
    Ae = A[i, j, 2]
    Aw = A[i, j, 3]
    An = A[i, j, 4]
    As = A[i, j, 5]
    r  = b[i, j] - (Ac*xc + Ae*xe + Aw*xw + An*xn + As*xs)
    return r
end

function kernel_colored_sor_sweep!(A, x, b, ω, sz, gc, c)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        if (i + j)%2 == c
            x[i, j] += ω*cell_residual(A, x, b, i, j)/A[i, j, 1]
        end
    end
    return nothing
end

function kernel_residual!(A, x, b, r, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        r[i, j] = cell_residual(A, x, b, i, j)
    end
    return nothing
end

function gpu_sor!(A, x, b, r, ω, sz, gc, max_err, max_it, nthread)
    nblock = (
        cld(sz[1], nthread[1]),
        cld(sz[2], nthread[2])
    )
    it = 0
    err = 0
    while true
        @cuda threads=nthread blocks=nblock kernel_colored_sor_sweep!(
            A, x, b, ω,
            sz, gc, 0
        )
        @cuda threads=nthread blocks=nblock kernel_colored_sor_sweep!(
            A, x, b, ω,
            sz, gc, 1
        )
        @cuda threads=nthread blocks=nblock kernel_residual!(
            A, x, b, r,
            sz, gc
        )
        err = norm(r)
        ncell = prod(sz .- 2*gc)
        err = err/sqrt(ncell)
        it += 1
        if err <= max_err || it >= max_it
            break
        end
    end
    return it, err
end