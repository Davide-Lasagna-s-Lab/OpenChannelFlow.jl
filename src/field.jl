# Implementation of physical representation of scalar channel fields

struct Field{G<:ChannelGrid, T} <: AbstractScalarField{4, T}
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
    y, x, z, t = points(g, period, _padded_size((S[2], S[3], S[4]), Val(dealias)))
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

function _padded_size(shape::NTuple{3, Int}, ::Val{true})
    new_shape = zeros(Int, 3)
    for (i, s) in enumerate(shape)
        new_shape[i] = (3*s)>>1 + 1 - ((3*s)>>1)&1
    end
    return tuple(new_shape...)
end
_padded_size(sizes::NTuple{3, Int}, ::Val{false}) = sizes


# --------------------- #
# allocating transforms #
# --------------------- #
function FFT(u::Field{G, T}) where {S, G<:ChannelGrid{S}, T}
    û = FTField(grid(u), T)
    parent(û) .= rfft(parent(u), [2, 3, 4])
    û .*= 1/prod(S[2:4])
    return û
end

function FFT(u::Field{G, T}, N) where {S, G<:ChannelGrid{S}, T}
    û = growto(FTField(grid(u), rfft(parent(u), [2, 3, 4])./prod(S[2:4])), N)
    return û
end

FFT(u::VectorField{L, P})    where {L, P<:Field} = VectorField([FFT(u[n])    for n in 1:L]...)
FFT(u::VectorField{L, P}, N) where {L, P<:Field} = VectorField([FFT(u[n], N) for n in 1:L]...)

function IFFT(û::FTField{G, T}) where {S, G<:ChannelGrid{S}, T}
    u = Field(grid(û), T)
    parent(u) .= brfft(parent(û), S[2], [2, 3, 4])
    return u
end

function IFFT(û::FTField{G, T}, N) where {G, T}
    u = Field(growto(grid(û), N), T)
    parent(u) .= brfft(parent(growto(û, N)), N[1], [2, 3, 4])
    return u
end

IFFT(u::VectorField{L, S})    where {L, S<:FTField} = VectorField([IFFT(u[n])    for n in 1:L]...)
IFFT(u::VectorField{L, S}, N) where {L, S<:FTField} = VectorField([IFFT(u[n], N) for n in 1:L]...)
