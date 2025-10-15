# General broadcasting rules for both spectral and physical channel field types

# union field type
const ChannelField = Union{SCField, PCField, VectorField}

# basic array broadcasting behaviour
Base.BroadcastStyle(::Type{F}) where {F<:ChannelField} = Broadcast.ArrayStyle{F}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{F}}, ::Type{T}) where {F<:ChannelField, T} = similar(find_field(bc), T)

find_field(bc::Broadcast.Broadcasted) = find_field(bc.args)
find_field(args::Tuple)               = find_field(find_field(args[1]), Base.tail(args))
find_field(u::ChannelField, rest)     = u
find_field(::Any, rest)               = find_field(rest)
find_field(x)                         = x
find_field(::Tuple{})                 = nothing

# vector field broadcasting into underlying field
function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{F}}) where {N, F<:VectorField{N}}
    dest = similar(bc, eltype(find_field(bc)[1]))
    for i in 1:N
        copyto!(dest[i], unpack(bc, i))
    end
    return dest
end

function Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{F}}) where {N, F<:VectorField{N}}
    for i in 1:N
        copyto!(dest[i], unpack(bc, i))
    end
    return dest
end

@inline unpack(bc::Broadcast.Broadcasted, i) = Broadcast.Broadcasted(bc.f, _unpack(i, bc.args))
@inline unpack(x::Any,                    i) = x
@inline unpack(x::VectorField,            i) = x.elements[i]

@inline _unpack(i, args::Tuple)        = (unpack(args[1], i), _unpack(i, Base.tail(args))...)
@inline _unpack(i, args::Tuple{<:Any}) = (unpack(args[1], i),)
@inline _unpack(i, args::Tuple{})      = ()
