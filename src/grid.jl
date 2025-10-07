# Implementation of the RPCF grid

# TODO: generalise to include potential spatial independencies

struct ChannelGrid{S, D1<:AbstractMatrix, D2<:AbstractMatrix, PLANS}
    y::Vector{Float64}
    Dy::D1
    Dy2::D2
    ws::Vector{Float64}
    β::Float64
    plans::PLANS

    ChannelGrid{S}(y, Dy, Dy2, ws, β, plans) where {S} =
        new{S, typeof(Dy), typeof(Dy2), typeof(plans)}(y, Dy, Dy2, ws, β, plans)
end

# outer constructor
function ChannelGrid(y, Nz, Nt, β,
                     Dy::D1, Dy2::D2, ws;
                     dealias::Bool=false,
                     flags::UInt32=FFTW.EXHAUSTIVE,
                     timelimit::Real=FFTW.NO_TIMELIMIT) where {D1, D2}
        plans = FFTPlans(length(y), Nz, Nt, dealias=dealias, pad=3/2, flags=flags, timelimit=timelimit)
        return ChannelGrid{(length(y), Nz, Nt)}(y, Dy, Dy2, ws, β, plans)
    end

# get points from grid
points(g::ChannelGrid{S}, T, dealias::Bool=false) where {S} = dealias ? points(g, T, get_array_sizes(g.plans)) : points(g, T, (S[2], S[3]))
points(g::ChannelGrid, T, S::NTuple{2, Int}) = (g.y, (0:(S[1] - 1))/(S[1])*(2π/g.β), (0:(S[2] - 1))/(S[2])*T)

# grow grid size
# ! this method does not grow the size of the transforms, making them incompatible with new fields constructed from growto!
growto!(g::ChannelGrid{S}, N::NTuple{2, Int}) where {S} =
     ChannelGrid{(S[1], N...)}(g.y, g.Dy, g.Dy2, g.ws, g.β, g.plans)
