include("nipce_setup.jl")

γ = computeSP2(op)
U = zeros(N, N, degree+1)
V = zeros(N, N, degree+1)
P = zeros(N, N, degree+1)

println("γ = $γ")

for sampleId=1:nsample
    filename = "data/nipce_$sampleId.csv"
    df = CSV.read(filename, DataFrame)
    println("read $filename")
    u = reshape(df[!, "u"], (N,N))
    v = reshape(df[!, "v"], (N,N))
    p = reshape(df[!, "p"], (N,N))
    ψ = evaluate(x[sampleId], op)
    for i=1:N, j=1:N
        for k=1:degree+1
            c = ψ[k]*w[sampleId]/γ[k]
            U[i, j, k] += u[i, j]*c
            V[i, j, k] += v[i, j]*c
            P[i, j, k] += p[i, j]*c
        end
    end
end

dft = CSV.read("data/nipce_1.csv", DataFrame)
x = dft[!, "x"]
y = dft[!, "y"]
z = dft[!, "z"]

unames = [Symbol("u$(k-1)") for k = 1:degree+1]
vnames = [Symbol("v$(k-1)") for k = 1:degree+1]
pnames = [Symbol("p$(k-1)") for k = 1:degree+1]
df = DataFrame(x=x, y=y, z=z)
for k=1:degree+1
    df[!, unames[k]] = vec(U[:,:,k])
end
for k=1:degree+1
    df[!, vnames[k]] = vec(V[:,:,k])
end
for k=1:degree+1
    df[!, pnames[k]] = vec(P[:,:,k])
end
CSV.write("data/nipce_quad_coefficients.csv", df)

uE = zeros(N, N)
uVar = zeros(N, N)
vE = zeros(N, N)
vVar = zeros(N, N)
pE = zeros(N, N)
pVar = zeros(N, N)
for i=1:N, j=1:N
    uE[i, j] = U[i, j, 1]
    vE[i, j] = V[i, j, 1]
    pE[i, j] = P[i, j, 1]
    for k=2:degree+1
        uVar[i, j] += γ[k]*U[i, j, k]^2
        vVar[i, j] += γ[k]*V[i, j, k]^2
        pVar[i, j] += γ[k]*P[i, j, k]^2
    end
end

df2 = DataFrame(
    "x" => x,
    "y" => y,
    "z" => z,
    "E[u]" => vec(uE),
    "Var[u]" => vec(uVar),
    "E[v]" => vec(vE),
    "Var[v]" => vec(vVar),
    "E[p]" => vec(pE),
    "Var[p]" => vec(pVar)
)
CSV.write("data/nipce-quad-statistics.csv", df2)