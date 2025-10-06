# Implementation of the spectral representation of scalar channel fields

# -------------- #
# spectral field #
# -------------- #
struct SCField{G<:ChannelGrid, T} <: AbstractArray{ComplexF64, 3}
    grid::G
    data::Array{Complex{T}, 3}

    function SCField(g::G, data::Array{Complex{T}, 3}) where {G, T}
        _apply_symmetry!(data)
        new{G, T}(g, data)
    end
end
SCField(g::G, ::Type{T}=Float64) where {S, G<:ChannelGrid{S}, T} = SCField(g, zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3]))


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

# array indexing
Base.@propagate_inbounds function Base.getindex(u::SCField, ny::Int, nz::Int, nt::Int)
    @boundscheck checkbounds(parent(u), ny, nz, nt)
    @inbounds val = parent(u)[ny, nz, nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::SCField, val, ny::Int, nz::Int, nt::Int)
    @boundscheck checkbounds(parent(u), ny, nz, nt)
    @inbounds parent(u)[ny, nz, nt] = val
    return val
end

# mode number idexing
Base.@propagate_inbounds function Base.getindex(u::SCField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nz, _nt = _convert_modenumber(n, S[3])
    @boundscheck checkbounds(u, ny, _nz, _nt)
    @inbounds val = u[ny, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::SCField{G}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nz, _nt = _convert_modenumber(n, S[3])
    @boundscheck checkbounds(u, ny, _nz, _nt)
    @inbounds u[ny, _nz, _nt] = val
    return val
end


# ------------- #
# utility stuff #
# ------------- #
grid(u::SCField) = u.grid


# TODO: modify these to work on channel field types
# TODO: add IFFT method for field type which works with points provided by grid

# function FFT!(u::SCField)
#     grid(u).plans(u.spectral_field, u.physical_field)
#     return u
# end

# function IFFT!(u::SCField)
#     grid(u).plans(u.physical_field, u.spectral_field)
#     return u
# end
