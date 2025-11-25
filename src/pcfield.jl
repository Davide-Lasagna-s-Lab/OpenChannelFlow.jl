# Implementation of physical representation of scalar channel fields

struct PCField{G<:ChannelGrid, T} <: AbstractArray{T, 4}
    grid::G
    data::Array{T, 4}

    function PCField(g::G, data::Array{T, 4}) where {G, T}
        all(isodd.(size(data)[2:4])) || throw(ArgumentError("data must be odd"))
        new{G, T}(g, data)
    end
end
PCField(g::G, ::Type{T}=Float64; dealias::Bool=false) where {S, G<:ChannelGrid{S}, T} = PCField(g, (y, x, z, t)->zero(T), 1.0, T, dealias=dealias)

# construct from function
function PCField(g::ChannelGrid{S}, fun, period::Real, ::Type{T}=Float64; dealias::Bool=false) where {S, T}
    pad = dealias ? 3/2 : 1
    y, x, z, t = points(g, period, _padded_size((S[2], S[3], S[4]), Val(pad)))
    data = fun.(reshape(y, :, 1, 1, 1), reshape(x, 1, :, 1, 1), reshape(z, 1, 1, :, 1), reshape(t, 1, 1, 1, :))
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

function _padded_size(sizes::NTuple{3, Int}, ::Val{F}) where {F}
    psizes = zeros(Int, 3)
    for i in 1:3
        psizes[i] = _size_up(sizes[i], F)
    end
    return tuple(psizes...)
end
_padded_size(sizes::NTuple{3, Int}, ::Val{1}) = sizes

@inline _size_up(N, F) = N == 1 ? 1 : (Np = ceil(Int, N*F); (Np - N) % 2 == 0 ? Np : Np + 1)
