using JSON

const P = 5

function read_M(P::Int, fname::String)::Array
    M_data = JSON.parsefile(fname)
    M = zeros(P + 1, P + 1, P + 1)
    for l in M_data
        a = l[1]
        b = l[2]
        c = l[3]
        m = l[4]
        M[begin + a, begin + b, begin + c] = m
    end
    return M
end

M = read_M(P, "M.json")

function compute_utopia_convection(ww, w, c, e, ee, u, dx)
    return (u*(- ee + 8*e - 8*w + ww) + abs(u)*(ee - 4*e + 6*c - 4*w + ww))/(12*dx)
end

function compute_convection_at_cell(K, U, val, M, dx, i, j)
    convection = 0
    for a in axes(M, 1)
        u = U[a, 1]
        v = U[a, 2]
        for b in axes(M, 2)
            valc  = val[i, j, b]
            vale  = val[i + 1, j, b]
            valee = val[i + 2, j, b]
            valw  = val[i - 1, j, b]
            valww = val[i - 2, j, b]
            valn  = val[i, j + 1, b]
            valnn = val[i, j + 2, b]
            vals  = val[i, j - 1, b]
            valss = val[i, j - 2, b]
            m = M[a, b, K]
            convection += m*compute_utopia_convection(valww, valw, valc, vale, valee, u, dx)
            convection += m*compute_utopia_convection(valss, vals, valc, valn, valnn, v, dx)
        end
    end
    return convection
end

function compute_diffusion_at_cell(K, val, viscosity, dx, i, j)
    valc = val[i, j, K]
    vale = val[i + 1, j, K]
    valw = val[i - 1, j, K]
    valn = val[i, j + 1, K]
    vals = val[i, j - 1, K]
    return viscosity*((vale - 2*valc + valw) + (valn - 2*valc + vals))/(dx^2)
end
