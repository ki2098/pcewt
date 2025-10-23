using PolyChaos

function prepare_tensors(P)
    op = GaussOrthoPoly(P, Nrec=100, addQuadrature=true)
    t2 = Tensor(2, op)
    t3 = Tensor(3, op)
    T2 = [t2.get([I, J]) for I=0:P, J=0:P]
    T3 = [t3.get([I, J, K]) for I=0:P, J=0:P, K=0:P]
    return T2, T3
end

