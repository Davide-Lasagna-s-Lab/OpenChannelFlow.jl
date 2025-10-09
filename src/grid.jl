# Implementation of the RPCF grid

struct ChannelGrid{S, D1<:AbstractMatrix, D2<:AbstractMatrix}
    y::Vector{Float64}
    Dy::D1
    Dy2::D2
    ws::Vector{Float64}
    β::Float64

    ChannelGrid(y, Nz, Nt, β, Dy::D1, Dy2::D2, ws, ::Type{T}=Float64) where {T, D1, D2} = 
        new{(length(y), Nz, Nt), typeof(Dy), typeof(Dy2)}(y, Dy, Dy2, ws, β)
end

# get points from grid
points(g::ChannelGrid{S}, T) where {S}       = (g.y, (0:(S[2] - 1))/(S[2])*(2π/g.β), (0:(S[3] - 1))/(S[3])*T)
points(g::ChannelGrid, T, S::NTuple{2, Int}) = (g.y, (0:(S[1] - 1))/(S[1])*(2π/g.β), (0:(S[2] - 1))/(S[2])*T)

# grow grid size
growto!(g::ChannelGrid, N::NTuple{2, Int}) = ChannelGrid(g.y, N..., g.β, g.Dy, g.Dy2, g.ws)
