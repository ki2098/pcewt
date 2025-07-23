function foo(a, scale)
    a .*= scale
end

A = rand(100, 100)
foo((@view A[:,1]), 10)
