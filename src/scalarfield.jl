# Implementation of the RPCF scalar field

# TODO: implement and test derivative methods

struct RPCFField{S, DM, DEALIAS, PLAN, IPLAN} <: AbstractScalarField{3, Float64}
    grid::RPCFGrid{S, DM, DEALIAS, PLAN, IPLAN}
    spectral_field::Array{3, ComplexF64}
    physical_field::Array{3, Float64}

    function RPCFField(g::RPCFGrid{S, DM, DEALIAS, PLAN, IPLAN}) where {S, DM, DEALIAS, PLAN, IPLAN}
        spectral_field = zeros(ComplexF64, S[1], (S[2] >> 1) + 1, S[3])
        physical_field = DEALIAS ? zeros(Float64, S[1], padded_size(S[2:3]...)...) : zeros(Float64, S...)
        new{S, DM, DEALIAS, PLAN, IPLAN}(g, spectral_field, physical_field)
    end
end


# ----------------- #
# interface methods #
# ----------------- #
ReSolverInterface.grid(u::RPCFField) = u.grid
ReSolverInterface.parent(u::RPCFField) = u.spectral_field
ReSolverInterface.similar(u::RPCFField{S, DM, DEALIAS}) where {S, DM, DEALIAS} = RPCFField(grid(u), dealias=DEALIAS)

function ReSolverInterface.mult!(uv::RPCFField{S, DM, DEALIAS}, u::RPCFField{S, DM, DEALIAS}, v::RPCFField{S, DM, DEALIAS}) where {S, DM, DEALIAS}
    FFT!(u); FFT!(v)
    uv.physical_field .= u.physical_field .* v.physical_field
    IFFT!(uv)
    return uv
end

function ReSolverInterface.dot(u::RPCFField{S}, v::RPCFField{S}) where {S}
    prod = 0.0

    # loop over top half plane exclusive of mean spanwise mode
    for nt in 1:S[3], nz in 2:((S[2] >> 1) + 1), ny in 1:S[1]
        prod += p.grid.ws[ny]*real(dot(p[ny, nz, nt], q[ny, nz, nt]))
    end

    # loop over positive temporal modes for mean spanwise mode
    for nt in 2:((S[3] >> 1) + 1), ny in 1:S[1]
        prod += p.grid.ws[ny]*real(dot(p[ny, 1, nt], q[ny, 1, nt]))
    end

    # evaluate mean component contribution
    for ny in 1:S[1]
        prod += 0.5*p.grid.ws[ny]*real(dot(p[ny, 1, 1], q[ny, 1, 1]))
    end

    return ((8π^2)/(grid(u).β*grid(u).ω))*prod
end


# --------------- #
# utility methods #
# --------------- #
FFT!(u::RPCFField) = (grid(u).plans(u.spectral_field, u.physical_field); return u)
IFFT!(u::RPCFField) = (grid(u).plans(u.physical_field, u.spectral_field); return u)


# ------------------ #
# derivative methods #
# ------------------ #
ddy!(dudy::RPCFField{S}, u::RPCFField{S}) where {S} = LinearAlgebra.mul!(dudy, grid(u).Dy, u)
d2dy2!(d2udy2::RPCFField{S}, u::RPCFField{S}) where {S} = LinearAlgebra.mul!(d2udy2, grid(u).Dy2, u)

function ddz!(dudz::RPCFField{S}, u::RPCFField{S}) where {S}
    β = grid(u).β

    # loop over spanwise modes multiplying by modifier
    @inbounds begin
        for nt in 1:S[3], nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
            dudz[ny, nz, nt] = (1im*(nz - 1)*β)*u[ny, nz, nt]
        end
    end

    return dudz
end

function d2dz2!(d2udz2::RPCFField{S}, u::RPCFField{S}) where {S}
    β = grid(u).β

    # loop over spanwise modes multiplying by modifier
    @inbounds begin
        for nt in 1:S[3], nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
            d2udz2[ny, nz, nt] = (-(((nz - 1)*β)^2))*u[ny, nz, nt]
        end
    end

    return d2udz2
end

function ReSolverInterface.ddt!(dudt::RPCFField{S}, u::RPCFField{S}) where {S}
    ω = grid(u).ω

    # loop over positive temporal modes multiplying by modifier
    @inbounds begin
        for nt in 1:((S[3] >> 1) + 1), nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
            dudt[ny, nz, nt] = (1im*(nt - 1)*ω)*u[ny, nz, nt]
        end
    end

    # loop over negative temporal modes multiplying by modifier
    if Nt > 1
        @inbounds begin
            for nt in ((S[3] >> 1) + 2):S[3], nz in 1:((S[2] >> 1) + 1), ny in 1:S[1]
                dudt[ny, nz, nt] = (1im*(nt - 1 - Nt)*ω)*u[ny, nz, nt]
            end
        end
    end

    return dudt
end
