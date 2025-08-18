#!/usr/bin/env julia

include("solve.jl")
using .PceCfd
using CSV

cd(@__DIR__)
println("now working in $(pwd())")

s, f = PceCfd.init("setup.json")
# flush(stdout)
@time begin
for step = 1:s.max_step
    try
        rms_divU = PceCfd.time_integral!(s)
        # if step % 1000 == 0
            println("step = $step, rms divU = $(round(rms_divU, digits=10))")
            flush(stdout)
        # end
    catch e
        println("Error occured: $(e)")
        break
    end
end
println()
end

PceCfd.write_csv(f, s)