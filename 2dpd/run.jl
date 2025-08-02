include("pdmodel.jl")

cd(@__DIR__)
println("now working in $(pwd())")

const gc = 2

setup_json = JSON.parsefile("setup.json")

domain_json = setup_json["domain"]
xmin = domain_json["x range"][1]
xmax = domain_json["x range"][2]
ymin = domain_json["y range"][1]
ymax = domain_json["y range"][2]
nx = domain_json["divisions"][1]
ny = domain_json["divisions"][2]

sz = (nx + 2*gc, ny + 2*gc)
dx = (xmax - xmin)/nx
dy = (ymax - ymin)/ny
@assert dx==dy "different cell size on x and y direction, dx=$dx, dy=$dy"

x = [dx*(i - gc - 0.5) + xmin for i = 1:sz[1]]
y = [dx*(j - gc - 0.5) + ymin for j = 1:sz[2]]

dfunc = zeros(sz)

prepare_dfunc!(setup_json["turbines"], x, y, dx, dfunc, sz)

heatmap(x, y, transpose(dfunc))