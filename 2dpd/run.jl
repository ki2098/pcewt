include("solve.jl")
using .PceCfd
using CSV

cd(@__DIR__)
println("now working in $(pwd())")

s = PceCfd.init("setup.json")
println()
for step = 1:s.max_step
    rms_divU = PceCfd.time_integral!(s)
    print("\rstep = $step, rms divU = $(round(rms_divU, digits=10))")
end
println()

PceCfd.write_csv("result.csv", s)