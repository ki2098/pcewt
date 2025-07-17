using JSON

function read_M(P, fname)
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

function compute_utopia_convection(ww, w, c, e, ee, u, dx)
    return (u*(- ee + 8*e - 8*w + ww) + abs(u)*(ee - 4*e + 6*c - 4*w + ww))/(12*dx)
end

