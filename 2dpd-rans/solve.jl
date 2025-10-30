using JSON, CSV, DataFrames

include("eq.jl")
include("rans.jl")
include("bc.jl")
include("pd.jl")

nthreads=(16,16)

struct Ls
    A
    b
    r
    ω
    maxdiag
    maxit
    maxerr
end

struct Cfd
    u
    v
    ut
    vt
    uu
    vv
    p
    k
    ω
    kt
    ωt
    nut
    divU
    dfunc
    uin
    kin
    ωin
    ls::Ls
    x
    y
    dx
    dy
    dt
    Re
    sz
    gc
    maxstep
end

function init(path)
    gc = 2
    init_json = JSON.parsefile(path)
    domain_json = init_json["domain"]
    xmin = domain_json["x range"][1]
    xmax = domain_json["x range"][2]
    ymin = domain_json["y range"][1]
    ymax = domain_json["y range"][2]
    nx = domain_json["divisions"][1]
    ny = domain_json["divisions"][2]
    sz = (nx + 2*gc, ny + 2*gc)
    dx = (xmax - xmin)/nx
    dy = (ymax - ymin)/ny
    println("DOMAIN INFO")
    println("\tgc = $gc")
    println("\tx range = [$xmin, $xmax]")
    println("\ty range = [$ymin, $ymax]")
    println("\tdivison = ($(sz[1]-2*gc) $(sz[2]-2*gc))")
    println("\tcell size = ($dx, $dy)")

    time_json = init_json["time"]
    maxtime = time_json["end"]
    dt = time_json["dt"]
    maxstep::Int = maxtime/dt
    println("TIME INFO")
    println("\tend time = $maxtime")
    println("\tdt = $dt")
    println("\ttotal steps = $maxstep")

    Re = setup_json["cfd"]["Re"]
    println("CFD INFO")
    println("\tRe = $Re")

    inlet_json = init_json["inlet"]
    uin = inlet_json["u"]
    kin = inlet_json["k"]
    ωin = inlet_json["omega"]
    println("INLET INFO")
    println("\tu = $uin")
    println("\tk = $kin")
    println("\tω = $ωin")

    A, b, r, maxdiag = gpu_init_pressure_eq(dx, dy, sz, gc, nthreads)
    ls = Ls(A, b, r, 1.2, maxdiag, 1000, 1e-4)
    println("EQ INFO")
    println("\tmax diag(A) = $maxdiag")

    x_h = [dx*(i - gc - 0.5) + xmin for i = 1:sz[1]]
    y_h = [dy*(j - gc - 0.5) + ymin for j = 1:sz[2]]
    x = CuArray(x_h)
    y = CuArray(y_h)
    u    = CUDA.zeros(Float64, sz...)
    v    = CUDA.zeros(Float64, sz...)
    ut   = CUDA.zeros(Float64, sz...)
    vt   = CUDA.zeros(Float64, sz...)
    uu   = CUDA.zeros(Float64, sz...)
    vv   = CUDA.zeros(Float64, sz...)
    p    = CUDA.zeros(Float64, sz...)
    k    = CUDA.zeros(Float64, sz...)
    ω    = CUDA.zeros(Float64, sz...)
    kt   = CUDA.zeros(Float64, sz...)
    ωt   = CUDA.zeros(Float64, sz...)
    nut  = CUDA.zeros(Float64, sz...)
    divU = CUDA.zeros(Float64, sz...)
    fill!(u , uin)
    fill!(ut, uin)
    fill!(uu, uin)
    fill!(k , kin)
    fill!(ω , ωin)
    dfunc = prepare_dfunc(init_json["wind turbines"], x, y, dx, dy, sz)
    

    cfd = Cfd(
        u, v, ut, vt, uu, vv, p,
        k, ω, kt, ωt, nut,
        divU, dfunc,
        uin, kin, ωin,
        ls,
        x, y, dx, dy, dt, Re,
        sz, gc, maxstep
    )

    output_path = init_json["output"]
    println("output to $output_path")

    return cfd
end