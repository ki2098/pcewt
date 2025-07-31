using NLsolve
using LinearAlgebra

function cell_forceK(uc, vc, Umagc, dfunc, T2, T3, K, P)
    fx, fy = 0.0, 0.0
    for I = 1:P+1
        for J = 1:P+1
            fx += uc[I]*Umagc[J]*T3[I, J, K]
            fy += vc[I]*Umagc[J]*T3[I, J, K]
        end
    end
    fx /= T2[K, K]
    fy /= T2[K, K]
    return fx, fy
end