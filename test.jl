function check(n)
    if n%2==0
        p = "even"
    else
        p = "odd"
    end
    return p
end

function cumulate(n)
    for i = 1:n
        b = i^2
        if b >= 10
            return b
        end
    end
end

println(cumulate(6))