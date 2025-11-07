include("nipce_setup.jl")

for sampleId=1:nsample
    ulid = x[sampleId]*σ + μ
    println("niPCE case $sampleId, ulid=$ulid")
    run_solver(ulid, L, N, Re, T, dt, "data/nipce_$sampleId.csv")
end

