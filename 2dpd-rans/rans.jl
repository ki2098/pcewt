include("cfd.jl")

const βasterisk = 0.09
const γ1 = 5.0/9
const β1 = 0.075
const σk1 = 0.85
const σω1 = 0.5
const γ2 = 0.44
const β2 = 0.0828
const σk2 = 1.0
const σω2 = 0.856

function kernel_kωSST_2003!(knew, ωnew, kold, ωold, u, v, uu, vv, nut, nu, z, dx, dy, sz, gc)
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    if gc < i <= sz[1]-gc && gc < j <= sz[2]-gc
        nutc = nut[i, j]
        kc = kold[i, j]
        ke = kold[i+1, j]
        kw = kold[i-1, j]
        kn = kold[i, j+1]
        ks = kold[i, j-1]
        ωc = ωold[i, j]
        ωe = ωold[i+1, j]
        ωw = ωold[i-1, j]
        ωn = ωold[i, j+1]
        ωs = ωold[i, j-1]
        dkdx = (ke - kw)/(2*dx)
        dkdy = (kn - ks)/(2*dy)
        dωdx = (ωe - ωw)/(2*dx)
        dωdy = (ωn - ωs)/(2*dy)
        CDkω = max(1e-10, 2*σω2*(dkdx*dωdx + dkdy*dωdy)/ωc)
        arg1 = 4*σω2*kc /(CDkω * z^2)
        arg2 = 500*nu / (z^2 * ωc)
        arg3 = sqrt(kc) / (βasterisk*ωc*z)
        arg4 = min(max(arg2, arg3), arg1)
        F1 = tanh(arg4^4)
        σk = F1*σk1 + (1 - F1)*σk2
        σω = F1*σω1 + (1 - F1)*σω2

        k_convection = cell_convection(kold, u, v, uu, vv, dx, dy, i, j)
        k_diffusion = cell_diffusion(kold, nu + nutc*σk, dx, dy, i, j)
        ω_convection = cell_convection(ωold, u, v, uu, vv, dx, dy, i, j)
        ω_diffusion = cell_diffusion(ωold, nu + nutc*σω, dx, dy, i, j)

        ue = u[i+1, j]
        uw = u[i-1, j]
        un = u[i, j+1]
        us = u[i, j-1]
        ve = v[i+1, j]
        vw = v[i-1, j]
        vn = v[i, j+1]
        vs = v[i, j-1]
        dudx = (ue - uw)/(2*dx)
        dudy = (un - us)/(2*dy)
        dvdx = (ve - vw)/(2*dx)
        dvdy = (vn - vs)/(2*dy)

        Pk = 2*dudx^2 + 2*dvdy^2 + dudy^2 + dvdx^2 + 2*dvdx*dudy
        Pktilde = min(Pk, 10*βasterisk*kc*ωc)
        Sk = βasterisk*kc*ωc

        γ = F1*γ1 + (1 - F1)*γ2
        β = F1*β1 + (1 - F1)*β2
        Pω = γ*Pktilde/nutc
        Sω = β*ωc^2
        kε_extra = 2*(1 - F1)*σω2*(dkdx*dωdx + dkdy*dωdy)/ωc

        kc += dt*(- k_convection + Pktilde - Sk + k_diffusion)
        ωc += dt*(- ω_convection + Pω - Sω + ω_diffusion + kε_extra)

        knew[i, j] = kc
        ωnew[i, j] = ωc

        arg5 = 500*nu / (z^2 * ωc)
        arg6 = 2*sqrt(kc) / (βasterisk*ωc*z)
        arg7 = max(arg5, arg6)
        F2 = tanh(arg7^2)
        S = 2*dudx^2 + 2*dvdy^2 + (dudy + dvdx)^2
        a1 = 0.31
        nutc = a1*kc / max(a1*ωc, S*F2)
        nut[i, j] = nutc
    end
    nothing
end

function gpu_kωSST_2003!(knew, ωnew, kold, ωold, u, v, uu, vv, nut, nu, z, dx, dy, sz, gc, nthread)
    nblock = (cld(sz[1], nthread[1]), cld(sz[2], nthread[2]))
    @cuda threads=nthread blocks=nblock kernel_kωSST_2003!(
        knew, ωnew, kold, ωold, u, v, uu, vv, nut, nu, z, dx, dy, sz, gc
    )
end