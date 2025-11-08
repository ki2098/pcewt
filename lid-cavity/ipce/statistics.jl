using CSV
using DataFrames
using Base.Threads

include("pcem.jl")

input_fname = ARGS[1]

input_df = CSV.read(input_fname, DataFrame)

column_names = names(input_df)

P = length(filter(name -> occursin(r"^u\d+$", name), column_names)) - 1

T2, _ = prepare_tensors(P)

u_names = [Symbol("u$(K-1)") for K = 1:P + 1]
v_names = [Symbol("v$(K-1)") for K = 1:P + 1]
p_names = [Symbol("p$(K-1)") for K = 1:P + 1]

row_cnt = nrow(input_df)

u_var = zeros(row_cnt)
v_var = zeros(row_cnt)
p_var = zeros(row_cnt)

@threads for i = 1:row_cnt
    for K = 2:P + 1
        uK = input_df[!, u_names[K]]
        u_var[i] += uK[i]^2 * T2[K, K]

        vK = input_df[!, v_names[K]]
        v_var[i] += vK[i]^2 * T2[K, K]

        pK = input_df[!, p_names[K]]
        p_var[i] += pK[i]^2 * T2[K, K]
    end
end

output_df = DataFrame(
    "x" => input_df.x,
    "y" => input_df.y,
    "z" => input_df.z,
    "E[u]" => input_df[!, u_names[1]],
    "Var[u]" => u_var,
    "E[v]" => input_df[!, v_names[1]],
    "Var[v]" => v_var,
    "E[p]" => input_df[!, p_names[1]],
    "Var[p]" => p_var
)

CSV.write("statistics.$input_fname", output_df)

