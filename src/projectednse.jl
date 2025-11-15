# Definitions for the projected versions of the Navier-Stokes equations that
# interface with the ReSolver.jl package.


# ----------------------- #
# projected navier-stokes #
# ----------------------- #
struct ProjectedNSE{T, NSE, LNSE, SF}
    nl::NSE
    ln::LNSE
    base::Vector{T}
    cache::NTuple{4, VectorField{3, SF}}

    function ProjectedNSE(g::ChannelGrid, Re::Real, ::Type{T}=Float64; Ro::Real=0, base::Vector{T}=g.y, flags=FFTW.EXHAUSTIVE) where {T}
        # construct operators
        plans = FFTPlans(g, T, dealias=true, flags=flags)
        scache = [VectorField(g, T, N=3, type=SCField)               for _ in 1:7]
        pcache = [VectorField(g, T, N=3, type=PCField, dealias=true) for _ in 1:6]
        nl = CouettePrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
        ln = CouettePrimitiveLNSE(T(Re), T(Ro), plans, scache, pcache)

        # construct independent cache
        cache = ntuple(_->VectorField(g, T, N=3, type=SCField), 4)

        new{T, typeof(nl), typeof(ln), eltype(cache[1])}(nl, ln, base, cache)
    end
end


# ----------------------- #
# operator call interface #
# ----------------------- #
function (eq::ProjectedNSE)(out::ProjectedField{G, M},
                              a::ProjectedField{G, M}) where {G, M}
    # aliases
    u   = eq.cache[1]
    N_u = eq.cache[3]

    # expand coefficients into spectral field
    expand!(u, a)
    add_base!(u, eq.base)

    # operator action
    eq.nl(0, u, N_u)

    # project result back onto basis
    project!(out, N_u)

    return out
end

function (eq::ProjectedNSE)(out::ProjectedField{G, M},
                               ::ProjectedField{G, M},
                              b::ProjectedField{G, M}) where {G, M}
    # aliases
    v    = eq.cache[2]
    M_uv = eq.cache[4]

    # expand coefficients into spectral fields
    expand!(v, b)

    # operator action
    eq.ln(0, v, M_uv, Val(true))

    # project result back onto basis
    project!(out, M_uv)

    return out
end

add_base!(u, base) = (u[1][:, 1, 1] .+= base; return u)
