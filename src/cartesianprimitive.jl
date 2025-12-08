# Operators for the primitive formulation of the Navier-Stokes equation for
# Couette type flows.

# ---------------------- #
# navier-stokes operator #
# ---------------------- #
mutable struct CartesianPrimitiveNSE{T, G, FFT} <: NSE{FTField{G, T}}
              Re::T
              Ro::T
     const plans::FFT
    const scache::Vector{VectorField{3, FTField{G, T}}}
    const pcache::Vector{VectorField{3, Field{G, T}}}

    function CartesianPrimitiveNSE(Re::T,
                                   Ro::T,
                                plans::FFT,
                               scache::Vector{VectorField{3, FTField{G, T}}},
                               pcache::Vector{VectorField{3, Field{G, T}}}) where {T, FFT, G}
        new{T, G, FFT}(Re, Ro, plans, scache, pcache)
    end
end

function CartesianPrimitiveNSE(g::ChannelGrid{S}, Re::Real, ::Type{T}=Float64; Ro::Real=0, flags=FFTW.EXHAUSTIVE) where {S, T}
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g, T)               for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, T, dealias=true) for _ in 1:3]...) for _ in 1:4]
    CartesianPrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
end

function (eq::CartesianPrimitiveNSE)(::Real,
                                    u::VectorField{3, <:FTField{G}},
                                  out::VectorField{3, <:FTField{G}}) where {G}
    # aliases
    dudx = eq.scache[1]
    dudy = eq.scache[2]
    dudz = eq.scache[3]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]

    # compute viscous term
    laplacian!(out, u)
    out .*= 1/eq.Re

    # compute derivatives
    ddx1!(dudx, u)
    ddx2!(dudy, u)
    ddx3!(dudz, u)

    # advection term
    eq.plans(U, u)
    eq.plans(dUdx, dudx)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(out, dUdx, true) # add the result to the output

    # coriolis term
    @. out[1] += eq.Ro*u[2]
    @. out[2] -= eq.Ro*u[1]

    return out
end


# --------------------------------- #
# linearised navier-stokes operator #
# --------------------------------- #
mutable struct CartesianPrimitiveLNSE{ADJ, T, G, FFT} <: LNSE{FTField{G, T}}
              Re::T
              Ro::T
     const plans::FFT
    const scache::Vector{VectorField{3, FTField{G, T}}}
    const pcache::Vector{VectorField{3, Field{G, T}}}

    function CartesianPrimitiveLNSE(Re::T,
                                    Ro::T,
                                 plans::FFT,
                                scache::Vector{VectorField{3, FTField{G, T}}},
                                pcache::Vector{VectorField{3, Field{G, T}}},
                               adjoint::Bool) where {T, FFT, G}
        new{adjoint, T, G, FFT}(Re, Ro, plans, scache, pcache)
    end
end

function CartesianPrimitiveLNSE(g::ChannelGrid{S}, Re::Real, ::Type{T}=Float64; Ro::Real=0, flags=FFTW.EXHAUSTIVE, adjoint=false) where {S, T}
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g, T)               for _ in 1:3]...) for _ in 1:6]
    pcache = [VectorField([  Field(g, T, dealias=true) for _ in 1:3]...) for _ in 1:8]
    CartesianPrimitiveLNSE(T(Re), T(Ro), plans, scache, pcache, adjoint)
end

function (eq::CartesianPrimitiveLNSE{ADJ})(::Real,
                                          u::VectorField{3, <:FTField{G}},
                                          v::VectorField{3, <:FTField{G}},
                                        out::VectorField{3, <:FTField{G}}) where {ADJ, G}
    # aliases
    dudx = eq.scache[1]
    dudy = eq.scache[2]
    dudz = eq.scache[3]
    U    = eq.pcache[1]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]

    # compute base derivatives
    ddx1!(dudx, u)
    ddx2!(dudy, u)
    ddx3!(dudz, u)

    # transform base field
    eq.plans(U, u)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)

    # compute the rest
    eq(0, v, out)

    return out
end

function (eq::CartesianPrimitiveLNSE{false})(::Real,
                                            v::VectorField{3, <:FTField{G}},
                                          out::VectorField{3, <:FTField{G}}) where {G}
    # aliases
    dudx = eq.scache[1]
    dvdx = eq.scache[4]
    dvdy = eq.scache[5]
    dvdz = eq.scache[6]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    V    = eq.pcache[5]
    dVdx = eq.pcache[6]
    dVdy = eq.pcache[7]
    dVdz = eq.pcache[8]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx1!(dvdx, v)
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(V, v)
    eq.plans(dUdx, dudx) # has to be recomputed since nonlinear equation overwrites this field with other data
    eq.plans(dVdx, dvdx)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdy[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n] # overwrite field to save memory
        @. dVdy[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(out, dVdy, true) # add the result to the output

    # coriolis term
    @. out[1] += eq.Ro*v[2]
    @. out[2] -= eq.Ro*v[1]

    return out
end

function (eq::CartesianPrimitiveLNSE{true})(::Real,
                                           v::VectorField{3, <:FTField{G}},
                                         out::VectorField{3, <:FTField{G}}) where {G}
    # aliases
    dudx = eq.scache[1]
    dvdx = eq.scache[4]
    dvdy = eq.scache[5]
    dvdz = eq.scache[6]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    V    = eq.pcache[5]
    dVdx = eq.pcache[6]
    dVdy = eq.pcache[7]
    dVdz = eq.pcache[8]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx1!(dvdx, v)
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(V, v)
    eq.plans(dUdx, dudx) # has to be recomputed since nonlinear equation overwrites this field with other data
    eq.plans(dVdx, dvdx)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdy[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n] # overwrite field to save memory
    end
    dVdz .*= 0.0
    for i in 1:3
        @. dVdz[1] -= V[i]*dUdx[i] # overwrite field to save memory
        @. dVdz[2] -= V[i]*dUdy[i] # overwrite field to save memory
        @. dVdz[3] -= V[i]*dUdz[i] # overwrite field to save memory
    end
    eq.plans(out, dVdy, true) # add the result to the output
    eq.plans(out, dVdz, true) # add the result to the output

    # coriolis term
    @. out[1] -= eq.Ro*v[2]
    @. out[2] += eq.Ro*v[1]

    return out
end

NSEBase.ndim(::Union{CartesianPrimitiveNSE, CartesianPrimitiveLNSE}) = 3
