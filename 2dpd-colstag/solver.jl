module PceWt

using CSV
using DataFrames

include("cfd.jl")
include("eq.jl")
include("pce.jl")
include("bc.jl")

struct Solver
    u
    v
    ut
    vt
    uu
    vv
    Umag
    divU
    uin_mean
    uin_sd
    Re
    A
    b
    r
    ω
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
    println("\tcell size = ($dx, $dy)")

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
    uin_mean = inlet_json["u mean"]
    uin_sd = inlet_json["u sd"]

    println("INLET INFO")
    println("\tu mean = $uin_mean")
    println("\tu sd = $uin_sd")

    x = [dx*(i - gc - 0.5) + xmin for i = 1:sz[1]]
    y = [dy*(j - gc - 0.5) + ymin for j = 1:sz[2]]

    T2, T3 = prepare_tensors(P)

    u = zeros(sz..., P + 1)
    v = zeros(sz..., P + 1)
    ut = zeros(sz..., P + 1)
    vt = zeros(sz..., P + 1)
    uu = zeros(sz..., P + 1)
    vv = zeros(sz..., P + 1)
    Umag = zeros(sz..., P + 1)
    divU = zeros(sz..., P + 1)
    p = zeros(sz..., P + 1)
    dfunc = zeros(sz)

    u[:, :, 1] .= uin_mean
    u[:, :, 2] .= uin_sd
    uu[:, :, 1] .= uin_mean
    uu[:, :, 2] .= uin_sd

    A, b, r, max_diag = init_pressure_eq(P, dx, dy, sz, gc)

    println("EQ INFO")
    println("\tmax diag(A) = $max_diag")

    prepare_dfunc!(setup_json["wind turbines"], x, y, dx, dy, dfunc, sz)

    output_path = setup_json["output"]

    println("output to $output_path")

    PD_Umag_init_guess!(u, v, Umag, sz)
    solve_PD_Umag!(u, v, Umag, dfunc, T3, P, sz, gc)

    return Solver(
        u, v, ut, vt, uu, vv,
        Umag, divU, uin_mean, uin_sd, Re,
        A, b, r, 1.5, max_diag, p, dfunc,
        x, y, dx, dy, sz, gc,
        max_time, max_step, dt,
        P, T2, T3
    ), output_path
end

function time_integral!(solver::Solver)
    solver.ut .= solver.u
    solver.vt .= solver.v

    pseudo_U!(
        solver.ut, solver.vt,
        solver.u, solver.v,
        solver.uu, solver.vv,
        solver.Umag, solver.dfunc,
        solver.T2, solver.T3, solver.P,
        1/solver.Re,
        solver.dx, solver.dy, solver.dt,
        solver.sz, solver.gc
    )
    interpol_UU!(
        solver.u, solver.v,
        solver.uu, solver.vv,
        solver.sz, solver.gc
    )
    apply_UUbc!(
        solver.uu, solver.vv,
        solver.uin_mean, solver.uin_sd,
        solver.P,
        solver.sz, solver.gc
    )
    pressure_eq_b!(
        solver.uu, solver.vv,
        solver.b, solver.P,
        solver.dx, solver.dy, solver.dt,
        solver.max_diag,
        solver.sz, solver.gc
    )
    solve_pressure_eq!(
        solver.A, solver.p, solver.b, solver.r, solver.ω,
        solver.sz, solver.gc, solver.P,
        1e-6, 1000
    )
    apply_pbc!(
        solver.p,
        solver.sz, solver.gc
    )
    update_U_by_gradp!(
        solver.u, solver.v,
        solver.uu, solver.vv,
        solver.p,
        solver.P,
        solver.dx, solver.dy, solver.dt,
        solver.sz, solver.gc
    )
    apply_Ubc!(
        solver.u, solver.v,
        solver.ut, solver.vt,
        solver.uin_mean, solver.uin_sd,
        solver.dx, solver.dy, solver.dt,
        solver.T2, solver.T3, solver.P,
        solver.sz, solver.gc
    )
    apply_UUbc!(
        solver.uu, solver.vv,
        solver.uin_mean, solver.uin_sd,
        solver.P,
        solver.sz, solver.gc
    )
    divU_mag = div_UK!(
        solver.uu, solver.vv, solver.divU,
        solver.dx, solver.dy, solver.P,
        solver.sz, solver.gc
    )
    if divU_mag > 1
        error("cfd solver failed to converge, |div(U)|/N = $divU_mag")
    end
    try
        solve_PD_Umag!(
            solver.u, solver.v,
            solver.Umag, solver.dfunc, 
            solver.T3, solver.P,
            solver.sz, solver.gc
        )
    catch pd_solver_error
        rethrow(pd_solver_error)
    end
    return divU_mag
end

function get_statistics(v, T2, P)
    var = 0.0
    for K = 2:P + 1
        var += v[K]^2 * T2[K, K]
    end
    return v[1], var
end

function write_csv(filename::String, solver::Solver)
    u = solver.u
    v = solver.v
    p = solver.p
    divU = solver.divU
    P = solver.P
    sz = solver.sz
    gc = solver.gc
    x = solver.x
    y = solver.y
    x_coord = zeros(sz)
    y_coord = zeros(sz)
    z_coord = zeros(sz)
    for j = 1:sz[2], i = 1:sz[1]
        x_coord[i, j] = x[i]
        y_coord[i, j] = y[j]
    end

    df = DataFrame(
        x = vec(@view x_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        y = vec(@view y_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc]),
        z = vec(@view z_coord[gc+1:sz[1]-gc, gc+1:sz[2]-gc])
    )
    u_names = [Symbol("u$(K-1)") for K = 1:P + 1]
    v_names = [Symbol("v$(K-1)") for K = 1:P + 1]
    # uu_names = [Symbol("uu$(K-1)") for K = 1:P + 1]
    # vv_names = [Symbol("vv$(K-1)") for K = 1:P + 1]
    p_names = [Symbol("p$(K-1)") for K = 1:P + 1]
    div_names = [Symbol("div$(K-1)") for K = 1:P + 1]

    for K = 1:P + 1
        df[!, u_names[K]] = vec(@view u[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K])
    end
    for K = 1:P + 1
        df[!, v_names[K]] = vec(@view v[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K])
    end
    # for K = 1:P + 1
    #     df[!, uu_names[K]] = vec(@view uu[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K])
    # end
    # for K = 1:P + 1
    #     df[!, vv_names[K]] = vec(@view vv[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K])
    # end
    for K = 1:P + 1
        df[!, p_names[K]] = vec(@view p[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K])
    end
    for K = 1:P + 1
        df[!, div_names[K]] = vec(@view divU[gc+1:sz[1]-gc, gc+1:sz[2]-gc, K])
    end

    CSV.write(filename, df)
    println("written to $filename")
end

end