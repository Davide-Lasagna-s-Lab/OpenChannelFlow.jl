# Operators for the primitive formulation of the Navier-Stokes equation for
# Couette type flows.

# ---------------------- #
# navier-stokes operator #
# ---------------------- #
mutable struct CouettePrimitiveNSE{T, SF, PF, FFT}
              Re::T
              Ro::T
     const plans::FFT
    const scache::Vector{VectorField{3, SF}}
    const pcache::Vector{VectorField{3, PF}}

    function CouettePrimitiveNSE(Re::T,
                                 Ro::T,
                              plans::FFT,
                             scache::Vector{VectorField{3, SF}},
                             pcache::Vector{VectorField{3, PF}}) where {T, FFT, SF, PF}
        new{T, SF, PF, FFT}(Re, Ro, plans, scache, pcache)
    end
end

function CouettePrimitiveNSE(g::ChannelGrid, Re::Real, ::Type{T}=Float64; Ro::Real=0, flags=FFTW.EXHAUSTIVE) where {T}
    plans = FFTPlans(g, T, dealias=true, flags=flags)
    scache = [VectorField(g, T, N=3, type=SCField)               for _ in 1:2]
    pcache = [VectorField(g, T, N=3, type=PCField, dealias=true) for _ in 1:3]
    CouettePrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
end

function (eq::CouettePrimitiveNSE)(::Real,
                                  u::VectorField{3, <:SCField{G}},
                                out::VectorField{3, <:SCField{G}}) where {G}
    # aliases
    dudy = eq.scache[1]
    dudz = eq.scache[2]
    U    = eq.pcache[1]
    dUdy = eq.pcache[2]
    dUdz = eq.pcache[3]

    # compute viscous term
    laplacian!(out, u)
    out .*= 1/eq.Re

    # compute derivatives
    ddx2!(dudy, u)
    ddx3!(dudz, u)

    # advection term
    eq.plans(U, u)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdy[n] = -U[2]*dUdy[n] - U[3]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(out, dUdy, Val(false)) # add the result to the output

    # coriolis term
    @. out[1] += eq.Ro*u[2]
    @. out[2] -= eq.Ro*u[1]

    return out
end


# --------------------------------- #
# linearised navier-stokes operator #
# --------------------------------- #
mutable struct CouettePrimitiveLNSE{T, SF, PF, FFT}
              Re::T
              Ro::T
     const plans::FFT
    const scache::Vector{VectorField{3, SF}}
    const pcache::Vector{VectorField{3, PF}}

    function CouettePrimitiveLNSE(Re::T,
                                  Ro::T,
                               plans::FFT,
                              scache::Vector{VectorField{3, SF}},
                              pcache::Vector{VectorField{3, PF}}) where {T, FFT, SF, PF}
        new{T, SF, PF, FFT}(Re, Ro, plans, scache, pcache)
    end
end

function CouettePrimitiveLNSE(g::ChannelGrid, Re::Real, ::Type{T}=Float64; Ro::Real=0, flags=FFTW.EXHAUSTIVE) where {T}
    plans = FFTPlans(g, T, dealias=true, flags=flags)
    scache = [VectorField(g, T, N=3, type=SCField)               for _ in 1:4]
    pcache = [VectorField(g, T, N=3, type=PCField, dealias=true) for _ in 1:6]
    CouettePrimitiveLNSE(T(Re), T(Ro), plans, scache, pcache)
end

function (eq::CouettePrimitiveLNSE)(::Real,
                                   u::VectorField{3, <:SCField{G}},
                                   v::VectorField{3, <:SCField{G}},
                                 out::VectorField{3, <:SCField{G}},
                             adjoint::Bool=false) where {G}
    # aliases
    dudy = eq.scache[1]
    dudz = eq.scache[2]
    U    = eq.pcache[1]
    dUdz = eq.pcache[3]

    # compute base derivatives
    ddx2!(dudy, u)
    ddx3!(dudz, u)

    # transform base field
    eq.plans(U, u)
    eq.plans(dUdz, dudz)

    # compute the rest
    eq(0, v, out, Val(adjoint))

    return out
end

function (eq::CouettePrimitiveLNSE)(::Real,
                                   v::VectorField{3, <:SCField{G}},
                                 out::VectorField{3, <:SCField{G}},
                                    ::Val{false}) where {G}
    # aliases
    dudy = eq.scache[1]
    dvdy = eq.scache[3]
    dvdz = eq.scache[4]
    U    = eq.pcache[1]
    dUdy = eq.pcache[2]
    dUdz = eq.pcache[3]
    V    = eq.pcache[4]
    dVdy = eq.pcache[5]
    dVdz = eq.pcache[6]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(V, v)
    eq.plans(dUdy, dudy)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdy[n]  = U[2]*dVdy[n] + U[3]*dVdz[n] # overwrite field to save memory
        @. dVdy[n] += V[2]*dUdy[n] + V[3]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(dvdy, dVdy) # overwrite field to save memory
    @. out -= dvdy

    # coriolis term
    @. out[1] += eq.Ro*v[2]
    @. out[2] -= eq.Ro*v[1]

    return out
end

function (eq::CouettePrimitiveLNSE)(::Real,
                                   v::VectorField{3, <:SCField{G}},
                                 out::VectorField{3, <:SCField{G}},
                                    ::Val{true}) where {G}
    # aliases
    dudy = eq.scache[1]
    dvdy = eq.scache[3]
    dvdz = eq.scache[4]
    U    = eq.pcache[1]
    dUdy = eq.pcache[2]
    dUdz = eq.pcache[3]
    V    = eq.pcache[4]
    dVdy = eq.pcache[5]
    dVdz = eq.pcache[6]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(V, v)
    eq.plans(dUdy, dudy)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdy[n] = U[2]*dVdy[n] + U[3]*dVdz[n] # overwrite field to save memory
    end
    dVdz .*= 0.0
    for i in 1:3
        @. dVdz[2] += V[i]*dUdy[i] # overwrite field to save memory
        @. dVdz[3] += V[i]*dUdz[i] # overwrite field to save memory
    end
    eq.plans(dvdy, dVdy) # overwrite field to save memory
    eq.plans(dvdz, dVdz) # overwrite field to save memory
    @. out += dvdy - dvdz

    # coriolis term
    @. out[1] -= eq.Ro*v[2]
    @. out[2] += eq.Ro*v[1]

    return out
end
