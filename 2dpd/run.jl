include("solve.jl")
using .PceCfd
using CSV

cd(@__DIR__)
println("now working in $(pwd())")

s, f = PceCfd.init("setup.json")
for step = 1:s.max_step
    try
        rms_divU = PceCfd.time_integral!(s)
        print("\rstep = $step, rms divU = $(round(rms_divU, digits=10))")
    catch e
        println("Error occured: $(e)")
        break
    end
end
println()

PceCfd.write_csv(f, s)