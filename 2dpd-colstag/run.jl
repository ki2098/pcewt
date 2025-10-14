include("solver.jl")

using Dates
using .PceWt

cd(@__DIR__)
println("now working in $(pwd())")

solver, output = PceWt.init("setup.json")

println(now())
for step = 1:solver.max_step
    # try
        div_err = PceWt.time_integral!(solver)
        println("It=$step, divU=$div_err    ")
        flush(stdout)
    # catch e
        # println("Error occured: $e")
        # break
    # end
end
println()
println(now())

PceWt.write_csv(output, solver)