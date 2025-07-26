using JSON
using CSV
using DataFrames

cd(@__DIR__)

msg_json = JSON.parsefile("sum.json")

sz = (msg_json["size"][1], msg_json["size"][2])
sample_count = msg_json["sample count"]

println("data generated at time = $(msg_json["time"])")
println("domain size = $sz")
println("sample count = $sample_count")

point_count = sz[1]*sz[2]
u_mean = zeros(point_count)
v_mean = zeros(point_count)
u_var = zeros(point_count)
v_var = zeros(point_count)
x_coord, y_coord, z_coord = nothing, nothing, nothing

for sample_id = 1:sample_count
    filename = "data/result_$(sample_id).csv"
    df = CSV.read(filename, DataFrame)
    u = df[!, "u"]
    v = df[!, "v"]
    du = u - u_mean
    dv = v - v_mean
    u_mean .+= du ./ sample_id
    v_mean .+= dv ./ sample_id
    du2 = u - u_mean
    dv2 = v - v_mean
    u_var .+= du .* du2
    v_var .+= dv .* dv2
    if sample_id == 1
        global x_coord = df[:, "x"]
        global y_coord = df[:, "y"]
        global z_coord = df[:, "z"]
    end
end
u_var ./= sample_count
v_var ./= sample_count


sum_df = DataFrame(x=x_coord, y=y_coord, z=z_coord, u_mean=u_mean, u_var=u_var, v_mean=v_mean, v_var=v_var)
CSV.write("data/statistics.csv", sum_df)