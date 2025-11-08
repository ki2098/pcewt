include("solve.jl")
include("basic_setup.jl")
include("mc_setup.jl")

sampleId = 1
while sampleId <= nSample
    ulid = x[sampleId]*σ + μ
    println("monte carlo case $sampleId, ulid=$ulid")
    try
        run_solver(ulid, L, N, Re, T, dt, "data/mc_$sampleId.csv")
    catch e
        println("error = $e, retry")
        continue
    end
    global sampleId += 1
end