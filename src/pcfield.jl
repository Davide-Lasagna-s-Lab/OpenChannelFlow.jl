# Implementation of physical representation of scalar channel fields

struct Field{G<:ChannelGrid, T} <: AbstractArray{T, 4}
    grid::G
    data::Array{T, 4}

    function Field(g::G, data::Array{T, 4}) where {G, T}
        all(isodd.(size(data)[2:4])) || throw(ArgumentError("data must be odd"))
        new{G, T}(g, data)
    end
end
Field(g::G, ::Type{T}=Float64; dealias::Bool=false) where {S, G<:ChannelGrid{S}, T} = Field(g, (y, x, z, t)->zero(T), 1.0, T, dealias=dealias)

# construct from function
function Field(g::ChannelGrid{S}, fun, period::Real, ::Type{T}=Float64; dealias::Bool=false) where {S, T}
    pad = dealias ? 3/2 : 1
    y, x, z, t = points(g, period, _padded_size((S[2], S[3], S[4]), Val(pad)))
    data = fun.(reshape(y, :, 1, 1, 1), reshape(x, 1, :, 1, 1), reshape(z, 1, 1, :, 1), reshape(t, 1, 1, 1, :))
    return Field(g, T.(data))
end


# ------------- #
# array methods #
# ------------- #
Base.parent(u::Field)                                 = u.data
Base.eltype(::Field{G, T}) where {G, T}               = T
Base.similar(u::Field, ::Type{T}=eltype(u)) where {T} = Field(grid(u), similar(parent(u), T))


# --------------- #
# utility methods #
# --------------- #
grid(u::Field) = u.grid

function _padded_size(sizes::NTuple{3, Int}, ::Val{F}) where {F}
    psizes = zeros(Int, 3)
    for i in 1:3
        psizes[i] = _size_up(sizes[i], F)
    end
    return tuple(psizes...)
end
_padded_size(sizes::NTuple{3, Int}, ::Val{1}) = sizes

@inline _size_up(N, F) = N == 1 ? 1 : (Np = ceil(Int, N*F); (Np - N) % 2 == 0 ? Np : Np + 1)
