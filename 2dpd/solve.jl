module PceCfd

include("cfd.jl")
include("eq.jl")
include("pcem.jl")
include("bc.jl")

struct Solver
    u
    v
    ut
    vt
    Umag
    divU
    uin_mean
    uin_sd
    Re
    A
    b
    max_diagA
    p
    dfunc
    x
    y
    dx
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

    println("DOMAIN INFO")
    println("\tgc = $gc")
    println("\tx range = [$xmin, $xmax]")
    println("\ty range = [$ymin, $ymax]")
    println("\tdivison = ($(sz[1]-2*gc) $(sz[2]-2*gc))")
    println("\tcell size = $dx")

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
    uin_mean = inlet_json["E[u]"]
    uin_sd = inlet_json["SD[u]"]

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
    Umag = zeros(sz..., P + 1)
    divU = zeros(sz..., P + 1)
    p = zeros(sz..., P + 1)
    dfunc = zeros(sz)

    u[:, :, 1] .= uin_mean
    u[:, :, 2] .= uin_sd

    A, b, max_diagA = init_pressure_eq(P, dx, sz, gc)

    println("EQ INFO")
    println("\tmax diag(A) = $max_diagA")

    prepare_dfunc!(setup_json["wind turbines"], x, y, dx, dfunc, sz)

    PD_Umag_init_guess!(u, v, Umag, sz)
    solve_PD_Umag!(u, v, Umag, dfunc, T3, P, sz, gc)

    return Solver(
        u, v, ut, vt, Umag, divU,
        uin_mean, uin_sd, Re,
        A, b, max_diagA, p,
        dfunc,
        x, y, dx, sz, gc,
        max_time, max_step, dt,
        P, T2, T3
    )
end

function time_integral!(s::Solver)
    s.ut .= s.u
    s.vt .= s.v

    pseudo_U!(
        s.ut, s.vt, s.u, s.v, s.Umag, s.dfunc,
        s.T2, s.T3, s.P,
        1.0/s.Re, s.dx, s.dt,
        s.sz, s.gc
    )
    pressure_eq_b!(
        s.u, s.v, s.b,
        s.P, s.dx, s.dt,
        s.max_diagA, s.sz, s.gc
    )
    solve_pressure_eq!(
        s.p, s.b, s.P
    )
    update_U_by_grad_p!(
        s.u, s.v, s.p,
        s.P, s.dx, s.dt,
        s.sz, s.gc
    )
    apply_U_bc!(
        s.u, s.v, s.ut, s.vt,
        s.uin_mean, s.uin_sd,
        s.dx, s.dt, s.T2, s.T3, s.P,
        s.sz, s.gc
    )
    solve_PD_Umag!(
        s.u, s.v, s.Umag, s.dfunc,
        s.T3, s.P,
        s.sz, s.gc
    )
    rms_divU = div_UK!(
        s.u, s.v, s.divU,
        s.dx, s.P, s.sz, s.gc
    )
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
    open(filename, "w") do f
        write(f, "x,y,z,E[u],Var[u],E[v],Var[v],E[p],Var[p],|div(U)|\n")
        gc = s.gc
        sz = s.sz
        x = s.x
        y = s.y
        u = s.u
        v = s.v
        p = s.p
        divU = s.divU
        T2 = s.T2
        P = s.P
        for j = gc+1:sz[2]-gc, i = gc+1:sz[1]-gc
            Eu, Vu = get_statistics(u[i, j, :], T2, P)
            Ev, Vv = get_statistics(v[i, j, :], T2, P)
            Ep, Vp = get_statistics(p[i, j, :], T2, P)
            write(f, "$(x[i]),$(y[j]),0,$Eu,$Vu,$Ev,$Vv,$Ep,$Vp,$(norm(divU[i,j,:]))\n")
        end
    end
end

end