using LinearAlgebra
using CUDA

function kernel_pressure_eq_A!(A, dx, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        Ae = Aw = An = As = 1/(dx^2)
        Ac = - (Ae + Aw + An + As)
        A[i, j, 1] = Ac
        A[i, j, 2] = Ae
        A[i, j, 3] = Aw
        A[i, j, 4] = An
        A[i, j, 5] = As
    end
    return nothing
end

function gpu_pressure_eq_A(dx, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    A = CUDA.zeros(Float64, sz..., 5)
    A[:, :, 1] .= 1
    @cuda threads=nthread blocks=nblock kernel_pressure_eq_A!(
        A, dx, sz, gc
    )
    maxdiag = maximum(abs.(A[:, :, 1]))
    A ./= maxdiag
    return A, maxdiag
end

function gpu_init_pressure_eq(P, dx, sz, gc, nthread)
    A, maxdiag = gpu_pressure_eq_A(dx, sz, gc, nthread)
    b = CUDA.zeros(Float64, sz..., P+1)
    r = CUDA.zeros(Float64, sz..., P+1)
    return A, b, r, maxdiag
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

function kernel_residual!(A, x, b, r, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        r[i, j] = cell_residual(A, x, b, i, j)
    end
    return nothing
end

function kernel_colored_sor_sweep!(A, x, b, ω, sz, gc, c)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc+1 <= i <= sz[1]-gc && gc+1 <= j <= sz[2]-gc
        if (i + j)%2 == c
            x[i, j] += ω*cell_residual(A, x, b, i, j)/A[i, j, 1]
        end
    end
    return nothing
end

function gpu_sor!(A, x, b, r, ω, sz, gc, maxerr, maxit, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    it = 0
    err = 0
    while true
        @cuda threads=nthread blocks=nblock kernel_colored_sor_sweep!(
            A, x, b, ω, sz, gc, 0
        )
        @cuda threads=nthread blocks=nblock kernel_colored_sor_sweep!(
            A, x, b, ω, sz, gc, 1
        )
        @cuda threads=nthread blocks=nblock kernel_residual!(
            A, x, b, r, sz, gc
        )
        err = norm(r)
        ncell = prod(sz .- 2*gc)
        err = err / sqrt(ncell)
        it += 1
        if err <= maxerr || it >= maxit
            break
        end
    end
    return it, err
end

function gpu_solve_eq!(A, p, b, r, ω, P, sz, gc, maxerr, maxit, nthread)
    for K = 1:P+1
        pK = @view p[:, :, K]
        bK = @view b[:, :, K]
        rK = @view r[:, :, K]
        gpu_sor!(A, pK, bK, rK, ω, sz, gc, maxerr, maxit, nthread)
    end
end

# checked