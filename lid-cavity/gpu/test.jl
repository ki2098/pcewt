using CUDA
using Cthulhu

function kernel_foo!(A, B, sz)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if i <= sz[1] && j <= sz[2]
        A[i, j] = B[i, j]
    end
    nothing
end

sz = (5, 5)

A = CUDA.rand(sz..., 5)
B = CUDA.rand(sz..., 5)

nthread = (5, 5)
nblock = (1, 1)

@cuda threads=nthread blocks=nblock kernel_foo!((@view A[:, :, 1]), (@view B[:, :, 2]), sz)


