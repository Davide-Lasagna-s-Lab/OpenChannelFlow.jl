# Explicit and implicit flow operators for the nonlinear and linearised
# Navier-Stokes equations.

# ----------------- #
# implicit operator #
# ----------------- #
mutable struct ImplicitOperator{T}
    Re::T

    ImplicitOperator(Re::Real, ::Type{T}=Float64) where {T} = new{T}(T(Re))
end

function (op::ImplicitOperator)(::Real, u::VectorField, out::VectorField)
    laplacian!(out, u)
    out .*= 1/op.Re
    return out
end


# ----------------- #
# explicit operator #
# ----------------- #
mutable struct ExplicitOperator{SF, PF, FFT, T}
              Ro::T
     const plans::FFT
    const scache::NTuple{2, VectorField{3, SF}}
    const pcache::NTuple{3, VectorField{3, PF}}

    function ExplicitOperator(g::G, Ro::Real, ::Type{T}=Float64, flags=FFTW.EXHAUSTIVE) where {G, T}
        scache = ntuple(_->VectorField(g, T, N=3, type=SCField), 2)
        pcache = ntuple(_->VectorField(g, T, N=3, type=PCField, dealias=true), 3)
        plans = FFTPlans(g, T, dealias=true, flags=flags)
        new{eltype(scache[1]), eltype(pcache[1]), typeof(plans), T}(T(Ro), plans, scache, pcache)
    end
end

# operator for ReSolver.jl
function (op::ExplicitOperator{SF})(::Real,
                                   u::VectorField{3, SF},
                                 out::VectorField{3, SF};
                                 add::Bool=false) where {SF}
    # aliases
    dudy = op.scache[1]
    dudz = op.scache[2]
    U    = op.pcache[1]
    dUdy = op.pcache[2]
    dUdz = op.pcache[3]

    # compute derivatives
    ddy!(dudy, u)
    ddz!(dudz, u)

    # advection term
    op.plans(U, u)
    op.plans(dUdy, dudy)
    op.plans(dUdz, dudz)
    for n in 1:3
        @. dUdy[n] = U[2]*dUdy[n] + U[3]*dUdz[n] # overwrite field that isn't needed anymore
    end
    if add
        out .-= op.plans(dudy, dUdy) # overwrite field that isn't needed anymore
    else
        out .= .-op.plans(dudy, dUdy) # overwrite field that isn't needed anymore
    end

    # coriolis term
    out[1] .+= op.Ro.*u[2]
    out[2] .-= op.Ro.*u[1]

    return out
end

# operator for Flows.jl
function (op::ExplicitOperator{SF})(::Real,
                                   q::VectorField{2, SF},
                                 out::VectorField{2, SF},
                                 add::Bool=false) where {SF}
    throw(error("Not implemented"))
end
