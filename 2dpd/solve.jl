module PceWt2d

using CSV
using DataFrames

include("cfd.jl")
include("bc.jl")
include("eq.jl")
include("pce.jl")
include("pd.jl")

struct Solver
    u
    v
    ut
    vt
    uu
    vv
    Umag
    divU
    Euin
    SDuin
    Re
    A
    b
    max_diag
    p
    dfunc
    x
    y
    dx
    dy
    sz
    gc
    max_time
    max_step
    dt
    P
    T2
    T3
end

function init(setup_filename)
    gc = 2
    setup_json = JSON.parsefile(setup_filename)

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

    println("DOMAIN INFO")
    println("\tgc = $gc")
    println("\tx range = [$xmin, $xmax]")
    println("\ty range = [$ymin, $ymax]")
    println("\tdivison = ($(sz[1]-2*gc) $(sz[2]-2*gc))")
    println("\tcell size = ($dx $dy)")

    time_json = setup_json["time"]
    max_time = time_json["end"]
    dt = time_json["dt"]
    max_step::Int = max_time/dt

    println("TIME INFO")
    println("\tend time = $max_time")
    println("\tdt = $dt")
    println("\ttotal steps = $max_step")

    Re = setup_json["cfd"]["Re"]
    P = setup_json["pce"]["P"]

    println("CFD INFO")
    println("\tRe = $Re")
    println("PCE INFO")
    println("\tP = $P")

    inlet_json = setup_json["inlet"]
    Euin = inlet_json["E[u]"]
    SDuin = inlet_json["SD[u]"]

    println("INLET INFO")
    println("\tE[u] = $uin_mean")
    println("\tSD[u] = $uin_sd")

    x = [dx*(i - gc - 0.5) + xmin for i = 1:sz[1]]
    y = [dx*(j - gc - 0.5) + ymin for j = 1:sz[2]]

    T2, T3 = prepare_tensors(P)

    u = zeros(sz..., P + 1)
    v = zeros(sz..., P + 1)
    ut = zeros(sz..., P + 1)
    vt = zeros(sz..., P + 1)
    uu = zeros(sz..., P + 1)
    vv = zeros(sz..., p + 1)
    Umag = zeros(sz..., P + 1)
    divU = zeros(sz..., P + 1)
    p = zeros(sz..., P + 1)
    dfunc = zeros(sz)

    u[:, :, 1] .= uin_mean
    u[:, :, 2] .= uin_sd
    uu .= u
    apply_UUbc!(uu, vv, Euin, SDuin, sz, gc)

    A, b, max_diag = init_pressure_eq(P, dx, dy, sz, gc)

    println("EQ INFO")
    println("\tmax diag(A) = $max_diag")

    prepare_dfunc!(setup_json["wind turbines"], x, y, dx, dy, dfunc, sz)

    output_path = setup_json["output"]

    println("output to $output_path")

    PD_Umag_init_guess!(u, v, Umag, sz)
    solve_PD_Umag!(u, v, Umag, dfunc, T3, P, sz, gc)

    return Solver(u, v, ut, vt, uu, vv, Umag, divU, Euin, SDuin, Re, A, b, max_diag, p, dfunc, x, y, dx, dy, sz, gc, max_time, max_step, dt, P, T2, T3)
end

function time_integral!(s::Solver)
    s.ut .= s.u
    s.vt .= s.v

    pseudo_U!(s.ut, s.vt, s.u, s.v, s.uu, s.vv, s.Umag, s.dfunc, s.T2, s.T3, s.P, 1.0/s.Re, s.dx, s.dy, s.dt, s.sz, s.gc)
    interpolate_UU!(s.u, s.v, s.uu, s.vv, s.sz, s.gc)
    apply_UUbc!(s.uu, s.vv, s.Euin, s.SDuin, s.sz, s.gc)
    pressure_eq_b!(s.uu, s.vv, s.b, s.P, s.dx, s.dy, s.dt, s.max_diag, s.sz, s.gc)
    solve_pressure_eq!(s.p, s.b, s.P)
    update_U_by_gradp!(s.u, s.v, s.uu, s.vv, s.p, s.P, s.dx, s.dy, s.dt, s.sz, s.gc)
    apply_Ubc!(s.u, s.v, s.ut, s.vt, s.Euin, s.SDuin, s.dx, s.dt, s.T2, s.T3, s.P, s.sz, s.gc)
    apply_UUbc!(s.uu, s.vv, s.Euin, s.SDuin, s.sz, s.gc)
    rms_divU = div_UK!(s.uu, s.vv, s.divU, s.dx, s.dy, s.P, s.sz, s.gc)
    try
        solve_PD_Umag!(s.u, s.v, s.Umag, s.dfunc, s.T3, s.P, s.sz, s.gc)
    catch e
        rethrow(e)
    end

    return rms_divU
end

function get_statistics(v, T2, P)
    var = 0.0
    for K = 2:P + 1
        var += v[K]^2 * T2[K, K]
    end
    return v[1], var
end

function write_csv(filename::String, s::Solver)
    u = s.u
    v = s.v
    p = s.p
    P = s.P
    sz = s.sz
    x = s.x
    y = s.y
    x_coord = zeros(sz)
    y_coord = zeros(sz)
    z_coord = zeros(sz)
    for j = 1:sz[2], i = 1:sz[1]
        x_coord[i, j] = x[i]
        y_coord[i, j] = y[j]
    end
    
    df = DataFrame(x = vec(x_coord), y = vec(y_coord), z = vec(z_coord))
    u_names = [Symbol("u$(K-1)") for K = 1:P + 1]
    v_names = [Symbol("v$(K-1)") for K = 1:P + 1]
    p_names = [Symbol("p$(K-1)") for K = 1:P + 1]

    for K = 1:P + 1
        df[!, u_names[K]] = vec(@view u[:, :, K])
    end
    for K = 1:P + 1
        df[!, v_names[K]] = vec(@view v[:, :, K])
    end
    for K = 1:P + 1
        df[!, p_names[K]] = vec(@view p[:, :, K])
    end

    CSV.write(filename, df)
end

end