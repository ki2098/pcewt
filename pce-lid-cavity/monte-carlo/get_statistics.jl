using JSON

old_dir = pwd()
cd(@__DIR__)

msg_json = JSON.parsefile("sum.json")

sz = (msg_json["size"][1], msg_json["size"][2])
sample_count = msg_json["sample count"]

println("data generated at time = $(msg_json["time"])")
println("domain size = $sz")
println("sample count = $sample_count")

cd(old_dir)