# Implementation of the RPCF scalar field

struct RPCFField{S, DM, DEALIAS, PAD, PLAN, IPLAN} <: AbstractScalarField{3, ComplexF64}
    grid::RPCFGrid{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    spectral_field::Array{ComplexF64, 3}
    physical_field::Array{Float64, 3}

    function RPCFField(g::RPCFGrid{S, DM, DEALIAS, PAD, PLAN, IPLAN}) where {S, DM, DEALIAS, PAD, PLAN, IPLAN}
        spectral_field = zeros(ComplexF64, S[1], (S[2] >> 1) + 1, S[3])
        physical_field = DEALIAS ? zeros(Float64, S[1], padded_size(S[2:3]..., PAD)...) : zeros(Float64, S...)
        new{S, DM, DEALIAS, PAD, PLAN, IPLAN}(g, spectral_field, physical_field)
    end
end


# ----------------- #
# interface methods #
# ----------------- #
ReSolverInterface.grid(u::RPCFField) = u.grid
ReSolverInterface.parent(u::RPCFField) = u.spectral_field
ReSolverInterface.similar(u::RPCFField{S, DM, DEALIAS}, ::Type{T}) where {S, DM, DEALIAS, T} = RPCFField(grid(u))

function mult_add!(uv::RPCFField{S, DM, DEALIAS}, u::RPCFField{S, DM, DEALIAS}, v::RPCFField{S, DM, DEALIAS}) where {S, DM, DEALIAS}
    IFFT!(u); IFFT!(v); IFFT!(uv)
    uv.physical_field .+= u.physical_field .* v.physical_field
    FFT!(uv)
    return uv
end
ReSolverInterface.mult!(uv::RPCFField{S, DM, DEALIAS}, u::RPCFField{S, DM, DEALIAS}, v::RPCFField{S, DM, DEALIAS}) where {S, DM, DEALIAS} = (uv .= 0.0; return mult_add!(uv, u, v))

function ReSolverInterface.dot(u::RPCFField{S}, v::RPCFField{S}) where {S}
    prod = 0.0

    # loop over top half plane exclusive of mean spanwise mode
    for nt in 1:S[3], nz in 2:((S[2] >> 1) + 1), ny in 1:S[1]
        prod += grid(u).ws[ny]*real(dot(u[ny, nz, nt], v[ny, nz, nt]))
    end

    # loop over positive temporal modes for mean spanwise mode
    for nt in 2:((S[3] >> 1) + 1), ny in 1:S[1]
        prod += grid(u).ws[ny]*real(dot(u[ny, 1, nt], v[ny, 1, nt]))
    end

    # evaluate mean component contribution
    for ny in 1:S[1]
        prod += 0.5*grid(u).ws[ny]*real(dot(u[ny, 1, 1], v[ny, 1, 1]))
    end

    return ((8π^2)/(grid(u).β*grid(u).ω))*prod
end

ReSolverInterface.include_base!(u::VectorField{3, <:RPCFField}, base::Vector{Float64}) = (u[1][:, 1, 1] .+= base; return u)


# ------------------ #
# projection methods #
# ------------------ #
ReSolverInterface.projectedField(::Type{G}, modes::Array{ComplexF64, 4}) where {S, G<:RPCFGrid{S}} = Array{ComplexF64, 3}(undef, size(modes, 2), (S[2] >> 1) + 1, S[3])
channel_int(u::AbstractVector, ws::AbstractVector, v::AbstractVector) = sum(ws[i]*dot(u[i], v[i]) for i in eachindex(u))
function ReSolverInterface.project!(a::ProjectedField{G, 3}, u::VectorField{D, F}) where {D, S, F<:RPCFField{S}, G}
    a .= 0.0
    for i in eachindex(u), nt in 1:S[3], nz in 1:((S[2] >> 1) + 1), n in axes(a, 1)
        a[n, nz, nt] += channel_int(@view(ReSolverInterface.modes(a)[(S[1]*(i - 1) + 1):S[1]*i, n, nz, nt]), grid(u).ws, @view(u[i][:, nz, nt]))
    end
    return a
end

function ReSolverInterface.expand!(u::VectorField{D, F}, a::ProjectedField{G, 3}) where {D, S, F<:RPCFField{S}, G}
    for i in eachindex(u), nt in 1:S[3], nz in 1:((S[2] >> 1) + 1)
        ReSolverInterface.mul!(@view(u[i][:, nz, nt]), @view(ReSolverInterface.modes(a)[(S[1]*(i - 1) + 1):S[1]*i, :, nz, nt]), @view(a[:, nz, nt]))
    end
    return u
end


# --------------- #
# utility methods #
# --------------- #
FFT!(u::RPCFField) = (grid(u).plans(u.spectral_field, u.physical_field); return u)
function FFT!(u::VectorField{N, <:RPCFField}) where {N}
    for i in 1:N
        FFT!(u[i])
    end
    return u
end

IFFT!(u::RPCFField) = (grid(u).plans(u.physical_field, u.spectral_field); return u)
function IFFT!(u::VectorField{N, <:RPCFField}) where {N}
    for i in 1:N
        IFFT!(u[i])
    end
    return u
end


# ------------------ #
# derivative methods #
# ------------------ #
ddy!(dudy::RPCFField{S}, u::RPCFField{S}) where {S} = ReSolverInterface.mul!(dudy, grid(u).Dy, u)
d2dy2!(d2udy2::RPCFField{S}, u::RPCFField{S}) where {S} = ReSolverInterface.mul!(d2udy2, grid(u).Dy2, u)

function ddz_add!(dudz::RPCFField{S}, u::RPCFField{S}) where {S}
    β = grid(u).β

    # loop over spanwise modes multiplying by modifier
    @inbounds begin
        for nt in 1:S[3], nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
            dudz[ny, nz, nt] += 1im*(nz - 1)*β*u[ny, nz, nt]
        end
    end

    return dudz
end
ddz!(dudz::RPCFField{S}, u::RPCFField{S}) where {S} = (dudz .= 0.0; return ddz_add!(dudz, u))

function d2dz2_add!(d2udz2::RPCFField{S}, u::RPCFField{S}) where {S}
    β = grid(u).β

    # loop over spanwise modes multiplying by modifier
    @inbounds begin
        for nt in 1:S[3], nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
            d2udz2[ny, nz, nt] += -(((nz - 1)*β)^2)*u[ny, nz, nt]
        end
    end

    return d2udz2
end
d2dz2!(d2udz2::RPCFField{S}, u::RPCFField{S}) where {S} = (d2udz2 .= 0.0; return d2dz2_add!(d2udz2, u))

function ReSolverInterface.ddt!(dudt::RPCFField{S}, u::RPCFField{S}) where {S}
    ω = grid(u).ω

    # loop over positive temporal modes multiplying by modifier
    @inbounds begin
        for nt in 1:((S[3] >> 1) + 1), nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
            dudt[ny, nz, nt] = 1im*(nt - 1)*ω*u[ny, nz, nt]
        end
    end

    # loop over negative temporal modes multiplying by modifier
    if S[3] > 1
        @inbounds begin
            for nt in ((S[3] >> 1) + 2):S[3], nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
                dudt[ny, nz, nt] = 1im*(nt - 1 - S[3])*ω*u[ny, nz, nt]
            end
        end
    end

    return dudt
end
