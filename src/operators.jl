# Explicit and implicit flow operators for the nonlinear and linearised
# Navier-Stokes equations.

# -------------------------- #
# couette primitive equation #
# -------------------------- #
mutable struct CouettePrimitiveNSE{T, SF, PF, FFT}
              Re::T
              Ro::T
      const base::Vector{T}
     const plans::FFT
    const scache::NTuple{4, VectorField{3, SF}}
    const pcache::NTuple{3, VectorField{3, PF}}

    function CouettePrimitiveNSE(g::ChannelGrid, Re::Real, ::Type{T}=Float64; Ro::Real=0, base::Vector{T}=g.y, flags=FFTW.EXHAUSTIVE) where {T}
        plans = FFTPlans(g, T, dealias=true, flags=flags)
        scache = ntuple(_->VectorField(g, T, N=3, type=SCField), 4)
        pcache = ntuple(_->VectorField(g, T, N=3, type=PCField, dealias=true), 3)
        new{T, eltype(scache[1]), eltype(pcache[1]), typeof(plans)}(T(Re), T(Ro), T.(base), plans, scache, pcache)
    end
end

function (eq::CouettePrimitiveNSE)(::Real,
                                  u::VectorField{3, <:SCField{G}},
                                out::VectorField{3, <:SCField{G}}) where {G}
    # aliases
    dudy = eq.scache[3]
    dudz = eq.scache[4]
    U    = eq.pcache[1]
    dUdy = eq.pcache[2]
    dUdz = eq.pcache[3]

    # compute viscous term
    laplacian!(out, u)
    out .*= 1/eq.Re

    # compute derivatives
    ddy!(dudy, u)
    ddz!(dudz, u)

    # advection term
    eq.plans(U, u)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdy[n] = U[2]*dUdy[n] + U[3]*dUdz[n] # overwrite field to save memory
    end
    out .-= eq.plans(dudy, dUdy) # overwrite field to save memory

    # coriolis term
    @. out[1] += eq.Ro*u[2]
    @. out[2] -= eq.Ro*u[1]

    return out
end

function (eq::CouettePrimitiveNSE)(out::ProjectedField{G, M},
                                     a::ProjectedField{G, M}) where {G, M}
    # aliases
    u   = eq.cache[1]
    N_u = eq.cache[2]

    # expand coefficients in spectral field
    expand!(u, a)
    u[1][:, 1, 1] .+= eq.base

    # compute effect of operator
    eq(0, u, N_u)

    # project result back onto basis
    project!(out, N_u)

    return out
end


# ---------------------------- #
# couette wall-normal equation #
# ---------------------------- #
struct CouetteWallNormalNSE{IT, ET, FT}
    implicit::IT
    explicit::ET
     forcing::FT

    function CouetteWallNormalNSE(g::ChannelGrid, Re::Real, ::Type{T}=Float64; Ro::Real=0, forcing=nothing, flags=FFTW.EXHAUSTIVE) where {T}
        implicit = ImplicitCouetteWallNormal(Re, T)
        explicit = ExplicitCouetteWallNormal(g, Ro, T, flags)
        new{typeof(implicit), typeof(explicit), typeof(forcing)}(implicit, explicit, forcing)
    end
end

function (eq::CouetteWallNormalNSE)(t::Real,
                                    q::VectorField{2, <:SCField{G}},
                                  out::VectorField{2, <:SCField{G}}) where {G}
    eq.implicit(t, q, out)
    eq.explicit(t, q, out, add=true)
    !isnothing(eq.forcing) && forcing(t, q, out)
    return out
end

function splitexim(eq::CouetteWallNormalNSE)
    function wrapper(t::Real,
                     q::VectorField{2, <:SCField{G}},
                   out::VectorField{2, <:SCField{G}}) where {G}
        eq.explicit(t, q, out, add=false)
        eq.forcing(t, q, out)
        return out
    end

    return wrapper, eq.implicit
end

mutable struct ImplicitCouetteWallNormal{T}
    Re::T

    ImplicitCouetteWallNormal(Re::Real, ::Type{T}=Float64) where {T} = new{T}(T(Re))
end

function (eq::ImplicitCouetteWallNormal)(::Real,
                                        q::VectorField{2, <:SCField{G}},
                                      out::VectorField{2, <:SCField{G}}) where {G}
    throw(error("some please implement me"))                             
end

mutable struct ExplicitCouetteWallNormal{T, SF, PF, FFT}
              Ro::T
     const plans::FFT
    const scache::NTuple{1, VectorField{2, SF}}
    const pcache::NTuple{1, VectorField{2, PF}}

    function ExplicitCouetteWallNormal(g::ChannelGrid, Ro::Real, ::Type{T}=Float64, flags=FFTW.EXHAUSTIVE) where {T}
        plans = FFTPlans(g, T, dealias=true, flags=flags)
        scache = ntuple(_->VectorField(g, T, N=3, type=SCField), 1)
        pcache = ntuple(_->VectorField(g, T, N=3, type=PCField, dealias=true), 1)
        new{T, eltype(scache[1]), eltype(pcache[1]), typeof(plans)}(T(Ro), plans, scache, pcache)
    end
end

function (eq::ExplicitCouetteWallNormal)(::Real,
                                        q::VectorField{2, <:SCField{G}},
                                      out::VectorField{2, <:SCField{G}}) where {G}
    throw(error("some please implement me"))                             
end
