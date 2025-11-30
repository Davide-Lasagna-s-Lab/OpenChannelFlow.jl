# Implementation of the spectral representation of scalar channel fields

# -------------- #
# spectral field #
# -------------- #
struct FTField{G<:ChannelGrid, T} <: AbstractScalarField{4, Complex{T}}
    grid::G
    data::Array{Complex{T}, 4}

    function FTField(g::G, data::Array{Complex{T}, 4}) where {G, T}
        apply_symmetry!(data)
        data[:, 1, 1, 1] .= real.(data[:, 1, 1, 1])
        new{G, T}(g, data)
    end
end
FTField(g::G, ::Type{T}=Float64) where {S, G<:ChannelGrid{S}, T} = FTField(g, zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3], S[4]))


# ------------- #
# array methods #
# ------------- #
Base.parent(u::FTField)                                 = u.data
Base.eltype(::FTField{G, T}) where {G, T}               = Complex{T}
Base.similar(u::FTField, ::Type{T}=eltype(u)) where {T} = FTField(grid(u), similar(parent(u), T))


# ---------------- #
# indexing methods #
# ---------------- #
# mode number idexing
Base.@propagate_inbounds function Base.getindex(u::FTField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(u[ny, _nx, _nz, _nt]) : u[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::FTField{G, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[4] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
                @inbounds u[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds u[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# --------------- #
# utility methods #
# --------------- #
grid(u::FTField) = u.grid
grid(u::VectorField) = grid(u[1])

function growto(u::FTField{G, T}, N::NTuple{3, Int}) where {S, G<:ChannelGrid{S}, T}
    out = FTField(growto(grid(u), N), T)
    for ny in 1:S[1], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[4] >> 1):(S[4] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end

function growto(u::VectorField{L, <:SCField}, N::NTuple{3, Int}) where {L}
    v = VectorField(growto(u[1], N), N=L)
    for n in 2:L
        parent(v[n]) .= parent(growto(u[n], N))
    end
    return v
end
