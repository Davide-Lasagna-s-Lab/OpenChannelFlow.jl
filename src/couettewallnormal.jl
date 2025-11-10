# Operators for the wall-normal formulation of the Navier-Stokes equation for
# Couette type flows.

# ---------------------- #
# navier-stokes operator #
# ---------------------- #
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
