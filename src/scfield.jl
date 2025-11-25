# Implementation of the spectral representation of scalar channel fields

# -------------- #
# spectral field #
# -------------- #
struct SCField{G<:ChannelGrid, T} <: AbstractArray{Complex{T}, 4}
    grid::G
    data::Array{Complex{T}, 4}

    function SCField(g::G, data::Array{Complex{T}, 4}) where {G, T}
        apply_symmetry!(data)
        data[:, 1, 1, 1] .= real.(data[:, 1, 1, 1])
        new{G, T}(g, data)
    end
end
SCField(g::G, ::Type{T}=Float64) where {S, G<:ChannelGrid{S}, T} = SCField(g, zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3], S[4]))


# ------------- #
# array methods #
# ------------- #
Base.IndexStyle(::Type{<:SCField})                      = Base.IndexLinear()
Base.parent(u::SCField)                                 = u.data
Base.eltype(::SCField{G, T}) where {G, T}               = Complex{T}
Base.size(u::SCField)                                   = size(parent(u))
Base.similar(u::SCField, ::Type{T}=eltype(u)) where {T} = SCField(grid(u), similar(parent(u), T))
Base.copy(u::SCField)                                   = SCField(grid(u), copy(parent(u)))
Base.zero(u::SCField)                                   = SCField(grid(u), zero(parent(u)))
Base.abs(u::SCField)                                    = (v = zero(u); v .= abs.(u); return v)


# ---------------- #
# indexing methods #
# ---------------- #
# linear indexing
Base.@propagate_inbounds function Base.getindex(u::SCField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds val = parent(u)[i]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::SCField, val, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = val
    return val
end

# mode number idexing
Base.@propagate_inbounds function Base.getindex(u::SCField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(u[ny, _nx, _nz, _nt]) : u[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::SCField{G, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}, T}
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
grid(u::SCField) = u.grid

function growto(u::SCField{G, T}, N::NTuple{3, Int}) where {S, G<:ChannelGrid{S}, T}
    out = SCField(growto(grid(u), N), T)
    for ny in 1:S[1], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[4] >> 1):(S[4] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end
