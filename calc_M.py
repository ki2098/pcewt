import chaospy

P = 5
normal = chaospy.Normal()
psi = chaospy.generate_expansion(P, normal)

for i in range(0, P+1):
    for j in range(0, P+1):
        for k in range(0, P+1):
            m = (chaospy.E(psi[i]*psi[j]*psi[k], normal)/chaospy.E(psi[k]*psi[k], normal)).round(5)
            if m != 0:
                print(f'{i} {j} {k} {m}')