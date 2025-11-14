# Operators for the primitive formulation of the Navier-Stokes equation for
# Couette type flows.

# TODO: create global constructor that allows re-use of cache objects

# types of operation for linear operator
struct TangentMode end
struct AdjointMode end
operator_mode(adjoint) = adjoint ? AdjointMode : TangentMode


# ---------------------- #
# navier-stokes operator #
# ---------------------- #
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
        new{T, eltype(scache[1]), eltype(pcache[1]), typeof(plans)}(T(Re), T(Ro), base, plans, scache, pcache)
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
    ddx2!(dudy, u)
    ddx3!(dudz, u)

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
    u   = eq.scache[1]
    N_u = eq.scache[2]

    # expand coefficients into spectral field
    expand!(u, a)
    u[1][:, 1, 1] .+= eq.base

    # operator action
    eq(0, u, N_u)

    # project result back onto basis
    project!(out, N_u)

    return out
end


# --------------------------------- #
# linearised navier-stokes operator #
# --------------------------------- #
mutable struct CouettePrimitiveLNSE{MODE, T, SF, PF, FFT}
              Re::T
              Ro::T
      const base::Vector{T}
     const plans::FFT
    const scache::NTuple{7, VectorField{3, SF}}
    const pcache::NTuple{6, VectorField{3, PF}}

    function CouettePrimitiveLNSE(g::ChannelGrid, Re::Real, ::Type{T}=Float64; Ro::Real=0, adjoint::Bool=false, base::Vector{T}=g.y, flags=FFTW.EXHAUSTIVE) where {T}
        plans = FFTPlans(g, T, dealias=true, flags=flags)
        scache = ntuple(_->VectorField(g, T, N=3, type=SCField), 7)
        pcache = ntuple(_->VectorField(g, T, N=3, type=PCField, dealias=true), 6)
        mode = operator_mode(adjoint)
        new{mode, T, eltype(scache[1]), eltype(pcache[1]), typeof(plans)}(T(Re), T(Ro), base, plans, scache, pcache)
    end
end

function (eq::CouettePrimitiveLNSE{TangentMode})(::Real,
                                                u::VectorField{3, <:SCField{G}},
                                                v::VectorField{3, <:SCField{G}},
                                              out::VectorField{3, <:SCField{G}}) where {G}
    # aliases
    dudy = eq.scache[4]
    dudz = eq.scache[5]
    dvdy = eq.scache[6]
    dvdz = eq.scache[7]
    U    = eq.pcache[1]
    V    = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    dVdy = eq.pcache[5]
    dVdz = eq.pcache[6]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx2!(dudy, u)
    ddx3!(dudz, u)
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(U, u)
    eq.plans(V, v)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for i in 1:3
        @. dVdy[i]  = U[2]*dVdy[i] + U[3]*dVdz[i] # overwrite field to save memory
        @. dVdy[i] += V[2]*dUdy[i] + V[3]*dUdz[i] # overwrite field to save memory
    end
    eq.plans(dvdy, dVdy) # overwrite field to save memory
    @. out -= dvdy

    # coriolis term
    @. out[1] += eq.Ro*v[2]
    @. out[2] -= eq.Ro*v[1]

    return out
end

function (eq::CouettePrimitiveLNSE{AdjointMode})(::Real,
                                                u::VectorField{3, <:SCField{G}},
                                                v::VectorField{3, <:SCField{G}},
                                              out::VectorField{3, <:SCField{G}}) where {G}
    # aliases
    dudy = eq.scache[4]
    dudz = eq.scache[5]
    dvdy = eq.scache[6]
    dvdz = eq.scache[7]
    U    = eq.pcache[1]
    V    = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    dVdy = eq.pcache[5]
    dVdz = eq.pcache[6]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx2!(dudy, u)
    ddx3!(dudz, u)
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(U, u)
    eq.plans(V, v)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for i in 1:3
        @. dVdy[i] = U[2]*dVdy[i] + U[3]*dVdz[i] # overwrite field to save memory
    end
    dVdz .*= 0.0
    for j in 1:3
        @. dVdz[2] += V[j]*dUdy[j] # overwrite field to save memory
        @. dVdz[3] += V[j]*dUdz[j] # overwrite field to save memory
    end
    eq.plans(dvdy, dVdy) # overwrite field to save memory
    eq.plans(dvdz, dVdz) # overwrite field to save memory
    @. out += dvdy - dvdz

    # coriolis term
    @. out[1] -= eq.Ro*v[2]
    @. out[2] += eq.Ro*v[1]

    return out
end

function (eq::CouettePrimitiveLNSE)(out::ProjectedField{G, M},
                                      a::ProjectedField{G, M},
                                      b::ProjectedField{G, M}) where {G, M}
    # aliases
    u    = eq.scache[1]
    v    = eq.scache[2]
    M_uv = eq.scache[3]

    # expand coefficients into spectral fields
    expand!(u, a)
    expand!(v, b)
    u[1][:, 1, 1] .+= eq.base

    # operator action
    eq(0, u, v, M_uv)

    # project result back onto basis
    project!(out, M_uv)

    return out
end
