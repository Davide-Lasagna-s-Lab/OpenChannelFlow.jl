# Nonlinear operator for the right-hand side of the Navier-Stokes equations.

# --------------------- #
# general operator type #
# --------------------- #
struct NonlinearOperator{IT, ET, FT, VF, T}
    implicit::IT
    explicit::ET
    forcing::FT
    base::Vector{T}
    cache::NTuple{2, VF}

    function NonlinearOperator(g::G, Re::Real, ::Type{T}=Float64; Ro::Real=0, base::Vector{T}=g.y, forcing=nothing, flags=FFTW.EXHAUSTIVE) where {G<:ChannelGrid, T}
        implicit = ImplicitOperator(Re, T)
        explicit = ExplicitOperator(g, Ro, T, flags)
        cache = ntuple(_->VectorField(g, T, N=3, type=SCField), 2)
        new{typeof(implicit), typeof(explicit), typeof(forcing), eltype(cache), T}(implicit, explicit, forcing, base, cache)
    end
end

function (eq::NonlinearOperator)(t::Real,
                                 q::VectorField{N, <:SCField{G}},
                               out::VectorField{N, <:SCField{G}}) where {N, G}
    eq.implicit(t, q, out)
    eq.explicit(t, q, out, add=true)
    !isnothing(eq.forcing) && eq.forcing(t, q, out)
    return out
end


# ------------------ #
# Flows.jl interface #
# ------------------ #
function splitexim(eq::NonlinearOperator)
    function wrapper(t::Real,
                     q::VectorField{2, <:SCField{G}},
                   out::VectorField{2, <:SCField{G}}) where {G}
        eq.explicit(t, q, out, add=false)
        eq.forcing(t, q, out)
        return out
    end

    return wrapper, eq.implicit
end


# --------------------- #
# ReSolver.jl interface #
# --------------------- #
function (eq::NonlinearOperator)(out::ProjectedField{G, M},
                                   a::ProjectedField{G, M}) where {G, M}
    u = eq.cache[1]
    N_u = eq.cache[2]
    expand!(u, a)
    u[1][:, 1, 1] .+= eq.base
    eq(0, u, N_u)
    project!(out, N_u)
    return out
end
