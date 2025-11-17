# Implementation of the RPCF grid

struct ChannelGrid{S, T, D1<:AbstractMatrix, D2<:AbstractMatrix}
    y::Vector{T}
    Dy::D1
    Dy2::D2
    ws::Vector{T}
    β::T

    function ChannelGrid(y, Nz, Nt, β, Dy::D1, Dy2::D2, ws, ::Type{T}=Float64) where {T, D1, D2}
        (isodd(Nz) && isodd(Nt)) || throw(ArgumentError("grid must be odd in spanwise and time directions"))
        new{(length(y), Nz, Nt), T, typeof(Dy), typeof(Dy2)}(T.(y), T.(Dy), T.(Dy2), T.(ws), T(β))
    end
end

# get points from grid
points(g::ChannelGrid{S}, T) where {S}       = (g.y, (0:(S[2] - 1))/(S[2])*(2π/g.β), (0:(S[3] - 1))/(S[3])*T)
points(g::ChannelGrid, T, S::NTuple{2, Int}) = (g.y, (0:(S[1] - 1))/(S[1])*(2π/g.β), (0:(S[2] - 1))/(S[2])*T)

# grow grid size
growto(g::ChannelGrid, N::NTuple{2, Int}) = ChannelGrid(g.y, N..., g.β, g.Dy, g.Dy2, g.ws)

# utility method to make mode generation easier with Resolvent.jl
get_fields(g::ChannelGrid) = (g.Dy, g.Dy2, g.ws, g.β)
