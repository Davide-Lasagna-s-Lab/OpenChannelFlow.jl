# Implementation of the RPCF grid

struct ChannelGrid{S, T, D1<:AbstractMatrix, D2<:AbstractMatrix}
    y::Vector{T}
    Dy::D1
    Dy2::D2
    ws::Vector{T}
    α::T
    β::T

    function ChannelGrid(y, Nx, Nz, Nt, α, β, Dy::D1, Dy2::D2, ws, ::Type{T}=Float64) where {T, D1, D2}
        (isodd(Nx) && isodd(Nz) && isodd(Nt)) || throw(ArgumentError("grid must be odd in spanwise and time directions"))
        new{(length(y), Nx, Nz, Nt), T, typeof(Dy), typeof(Dy2)}(T.(y), T.(Dy), T.(Dy2), T.(ws), T(α), T(β))
    end
end

# get points from grid
points(g::ChannelGrid{S}, T) where {S}       = (                           g.y,
                                                (0:(S[2] - 1))/(S[2])*(2π/g.α),
                                                (0:(S[3] - 1))/(S[3])*(2π/g.β),
                                                (0:(S[4] - 1))/(S[4])*T)
points(g::ChannelGrid, T, S::NTuple{3, Int}) = (                           g.y,
                                                (0:(S[1] - 1))/(S[1])*(2π/g.α),
                                                (0:(S[2] - 1))/(S[2])*(2π/g.β),
                                                (0:(S[3] - 1))/(S[3])*T)

# grow grid size
growto(g::ChannelGrid, N::NTuple{3, Int}) = ChannelGrid(g.y, N..., g.α, g.β, g.Dy, g.Dy2, g.ws)

# utility method to make mode generation easier with Resolvent.jl
get_fields(g::ChannelGrid) = (g.Dy, g.Dy2, g.ws, g.α, g.β)
