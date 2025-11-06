using CUDA
using Cthulhu

function bar(a, b)
    return a*sum(b)
end

function kernel_foo!(A, B, sz)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= 5 && j <= 5
        A[i, j, :] .= 2 .* ((@view B[i, j, :]) .+ (@view A[i, j, :]))
    end
    nothing
end

sz = (5, 5)

A = CUDA.ones(sz..., 5)
B = CUDA.rand(sz..., 5)

nthread = (5, 5)
nblock = (1, 1)

@cuda threads=nthread blocks=nblock kernel_foo!(A, B, sz)


