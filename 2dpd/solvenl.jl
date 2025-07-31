using PolyChaos
using NLsolve
using LinearAlgebra

P = 5
op = GaussOrthoPoly(P, Nrec=100, addQuadrature=true)
t3 = Tensor(3, op)
t2 = Tensor(2, op)
M = [t3.get([i, j, k])/t2.get([k, k]) for i=0:P, j=0:P, k=0:P]

b = [rand(), 0.05*rand(P)...]
c = [rand(), 0.05*rand(P)...]

function f!(r, a)
    for k = 1:P+1
        lsum = 0.0
        rsum = 0.0
        for i = 1:P+1
            for j = 1:P+1
                lsum += a[i]*a[j]*M[i, j, k]
                rsum +=  (b[i]*b[j] + c[i]*c[j])*M[i, j, k]
            end
        end
        r[k] = lsum - rsum
    end
end

function j!(J, a)
    fill!(J, 0.0)
    for k = 1:P+1
        for l = 1:P+1
            entry = 0.0
            for i = 1:P+1
                entry += a[i]*(M[l, i, k] + M[i, l, k])
            end
            J[k, l] = entry
        end
    end
end

a0 = sqrt.(b.^2 .+ c.^2)

sol = nlsolve(f!, j!, a0, ftol=1e-6)

println(sol)
r = zeros(P+1)
f!(r, sol.zero)
r