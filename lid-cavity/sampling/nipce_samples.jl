include("nipce_setup.jl")

for sampleId=1:nsample
    ulid = x[sampleId]*σ + μ
    println("niPCE case $sampleId, ulid=$ulid")
    run_solver(ulid, nu, L, N, T, dt, "data/nipce_$sampleId.csv")
end

