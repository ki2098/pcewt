import JSON

function read_M_from_file(P, filename)
    M = zeros(P + 1, P + 1, P + 1)
    Mjson = JSON.parsefile(filename)
    for (i, j, k, m) in Mjson
        M[i + 1, j + 1, k + 1] = m
    end
    return M
end
