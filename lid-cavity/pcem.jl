using PolyChaos

# function prepare_M(P)
#     op = GaussOrthoPoly(P, Nrec = 100, addQuadrature = true)
#     normsq = computeSP2(op)
#     t = Tensor(3, op)
#     T = [t.get([i, j, k]) for i=0:P, j=0:P, k=0:P]
#     M = zeros(P + 1, P + 1, P + 1)
#     for i = 1:P + 1, j = 1:P + 1, k = 1:P + 1
#         M[i, j, k] = round(T[i, j, k]/normsq[k], digits = 6)
#     end
#     return M, normsq
# end

# function show_M(M)
#     for i = axes(M, 1), j = axes(M, 2), k = axes(M, 3)
#         if M[i, j, k] != 0
#             println("$(i - 1) $(j - 1) $(k - 1) => $(M[i, j, k])")
#         end
#     end
# end

function prepare_tensors(P)
    op = GaussOrthoPoly(P, Nrec=100, addQuadrature=true)
    t2 = Tensor(2, op)
    t3 = Tensor(3, op)
    T2 = [t2.get([I, J]) for I=0:P, J=0:P]
    T3 = [t3.get([I, J, K]) for I=0:P, J=0:P, K=0:P]
    return T2, T3
end
