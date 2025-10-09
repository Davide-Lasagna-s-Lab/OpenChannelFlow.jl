# Implementation of physical representation of scalar channel fields

struct PCField{G<:ChannelGrid, T} <: AbstractArray{T, 3}
    grid::G
    data::Array{T, 3}

    function PCField(g::G, data::Array{T, 3}) where {G, T}
        all(isodd.(size(data)[2:3])) || throw(ArgumentError("data must be odd"))
        new{G, T}(g, data)
    end
end
PCField(g::G, ::Type{T}=Float64; dealias::Bool=false) where {S, G<:ChannelGrid{S}, T} = PCField(g, (y, z, t)->zero(T), 1.0, T, dealias=dealias)

# construct from function
function PCField(g::ChannelGrid{S}, fun, period::Real, ::Type{T}=Float64; dealias::Bool=false) where {S, T}
    pad = dealias ? 3/2 : 1
    y, z, t = points(g, period, _padded_size(S[2], S[3], Val(pad)))
    data = fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    return PCField(g, T.(data))
end


# ------------- #
# array methods #
# ------------- #
Base.IndexStyle(::Type{<:PCField})                      = IndexLinear()
Base.parent(u::PCField)                                 = u.data
Base.eltype(::PCField{G, T}) where {G, T}               = T
Base.size(u::PCField)                                   = size(parent(u))
Base.similar(u::PCField, ::Type{T}=eltype(u)) where {T} = PCField(grid(u), similar(parent(u), T))
Base.copy(u::PCField)                                   = PCField(grid(u), copy(parent(u)))
Base.zero(u::PCField)                                   = PCField(grid(u), zero(parent(u)))


# ---------------- #
# indexing methods #
# ---------------- #
# linear indexing
Base.@propagate_inbounds function Base.getindex(u::PCField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds val = parent(u)[i]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::PCField, val, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = val
    return val
end


# --------------- #
# utility methods #
# --------------- #
grid(u::PCField) = u.grid

function _padded_size(Nz, Nt, ::Val{F}) where {F}
    Nz_pad = ceil(Int, Nz*F)
    Nt_pad = ceil(Int, Nt*F)
    Nz_pad = (Nz_pad - Nz) % 2 == 0 ? Nz_pad : Nz_pad + 1
    Nt_pad = (Nt_pad - Nt) % 2 == 0 ? Nt_pad : Nt_pad + 1
    return Nz_pad, Nt_pad
end

_padded_size(Nz, Nt, ::Val{1}) = Nz, Nt
