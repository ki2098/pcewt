using JSON
using CSV
using ArgParse
using DataFrames

cd(@__DIR__)

s = ArgParseSettings()
@add_arg_table s begin
    "n"
        help="number of samples to be selected"
        arg_type=Int
        default=100
end
arg_parsed = parse_args(s)
n = arg_parsed["n"]

msg_json = JSON.parsefile("sum.json")

sz = (msg_json["size"][1], msg_json["size"][2])
sample_count = msg_json["sample count"]

println("data generated at time = $(msg_json["time"])")
println("domain size = $sz")
println("sample count = $sample_count")
println("samples to be selected = $n")