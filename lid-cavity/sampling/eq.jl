using LinearAlgebra
using CUDA

function kernel_pressure_eq_A!(A, dx, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        Ae = Aw = An = As = 1/(dx^2)
        Ac = - (Ae + Aw + An + As)
        A[i, j, 1] = Ac
        A[i, j, 2] = Ae
        A[i, j, 3] = Aw
        A[i, j, 4] = An
        A[i, j, 5] = As
    end
    nothing
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

function gpu_init_pressure_eq(dx, sz, gc, nthread)
    
end
