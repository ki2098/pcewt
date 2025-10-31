include("solve.jl")

using Dates

cd(@__DIR__)
println("now working in $(pwd())")

solver, output_path = init("setup.json")
println(now())
for step = 1:solver.maxstep
    lsit, lserr, divmag = time_integral!(solver)
    print("\rIt=$step, divU=$div_err     ")
    flush(stdout)
end
println()
println(now())

write_csv(output_path, solver)