# Vector field for generic channel fields

# ------------ #
# vector field #
# ------------ #
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    function VectorField(fields::Vararg{S, N}) where {S<:Union{PCField, SCField}, N}
        new{N, S}(fields)
    end
end

# construct using grid
VectorField(g::ChannelGrid, ::Type{T}=Float64; N::Int=3, type::Type{S}=SCField, kwargs...) where {T, S} = VectorField([type(g, T; kwargs...) for _ in 1:N]...)
VectorField(g::ChannelGrid, funcs, period, ::Type{T}=Float64; dealias::Bool=false) where {T} = VectorField([PCField(g, f, period, T, dealias=dealias) for f in funcs]...)


# ------------- #
# array methods #
# ------------- #
Base.IndexStyle(::Type{<:VectorField})                              = Base.IndexLinear()
Base.parent(q::VectorField)                                         = q.elements
Base.getindex(q::VectorField, i::Int)                               = parent(q)[i]
Base.size(::VectorField{N}) where {N}                               = (N,)
Base.eltype(::VectorField{N, F}) where {N, F}                       = F
datatype(q::VectorField)                                            = eltype(q.elements[1])
Base.similar(q::VectorField{N}, ::Type{T}=datatype(q)) where {N, T} = VectorField([similar(q.elements[n], T) for n in 1:N]...)
Base.copy(q::VectorField{N}) where {N}                              = VectorField([copy(q.elements[n]) for n in 1:N]...)
Base.zero(q::VectorField{N}) where {N}                              = VectorField([zero(q.elements[n]) for n in 1:N]...)


# --------------- #
# utility methods #
# --------------- #
grid(u::VectorField) = grid(u[1])

function growto(u::VectorField{L, <:SCField}, N::NTuple{3, Int}) where {L}
    v = VectorField(growto(grid(u), N), N=L, type=SCField)
    for n in 1:L
        parent(v[n]) .= parent(growto(u[n], N))
    end
    return v
end
