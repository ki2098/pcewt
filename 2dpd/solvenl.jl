using PolyChaos
using NLsolve
using LinearAlgebra

P = 5
op = GaussOrthoPoly(P, Nrec=100, addQuadrature=true)
t3 = Tensor(3, op)
t2 = Tensor(2, op)
M = [t3.get([i, j, k]) for i=0:P, j=0:P, k=0:P]

b = [rand(), 0.1*rand(P)...]
c = [rand(), 0.1*rand(P)...]

function f!(r, a)
    for k = 1:P+1
        lsum = 0.0
        rsum = 0.0
        for i = 1:P+1, j = 1:P+1
            lsum += a[i]*a[j]*M[i, j, k]
            rsum +=  (b[i]*b[j] + c[i]*c[j])*M[i, j, k]
        end
        r[k] = lsum - rsum
    end
end

function j!(J, a)
    fill!(J, 0.0)
    for k = 1:P+1, l = 1:P+1
        entry = 0.0
        for i = 1:P+1
            entry += a[i]*(M[l, i, k] + M[i, l, k])
        end
        J[k, l] = entry
    end
end

a0 = sqrt.(b.^2 .+ c.^2)

sol = nlsolve(f!, j!, a0, ftol=1e-6)

if NLsolve.converged(sol)
    sol.zero
else
    error("solver for U coefficients not converged\nfinal guess = $(sol.zero)\n")
end