# Implementation of the RPCF scalar field

# TODO: implement fourier transforms

struct RPCFField{S, DM, DEALIAS} <: AbstractScalarField{3, Float64}
    grid::RPCFGrid{S, DM}
    spectral_field::Array{3, ComplexF64}
    physical_field::Array{3, Float64}

    function RPCFField(g::RPCFGrid{S, DM}; dealias::Bool=true) where {S, DM}
        spectral_field = zeros(ComplexF64, S[1], (S[2] >> 1) + 1, S[3])
        physical_field = dealias ? zeros(Float64, S[1], padded_size(S[2:3]...)...) : zeros(Float64, S...)
        new{S, DM, dealias}(g, spectral_field, physical_field)
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


# ------------- #
# other methods #
# ------------- #
function padded_size(Nz, Nt) end

function FFT!(u::RPCFField) end
function IFFT!(u::RPCFField) end
