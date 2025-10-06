# Implementation of the RPCF grid

# TODO: generalise to include potential spatial independencies

struct ChannelGrid{S, D1<:AbstractMatrix, D2<:AbstractMatrix, PLANS}
    y::Vector{Float64}
    Dy::D1
    Dy2::D2
    ws::Vector{Float64}
    β::Float64
    plans::PLANS

    function ChannelGrid(y, Nz, Nt, β,
                         Dy::D1, Dy2::D2, ws;
                         dealias::Bool=true,
                         pad::Float64=3/2,
                         flags::UInt32=FFTW.EXHAUSTIVE,
                         timelimit::Real=FFTW.NO_TIMELIMIT) where {D1, D2}
        plans = FFTPlans(length(y), Nz, Nt, dealias=dealias, pad=pad, flags=flags, timelimit=timelimit)
        new{(length(y), Nz, Nt), D1, D2, typeof(plans)}(y, Dy, Dy2, ws, β, plans)
    end
end

# get points from grid
points(g::ChannelGrid{S}, T) where {S} = (g.y, (0:(S[2] - 1))/(S[2])*(2π/g.β), (0:(S[3] - 1))/(S[3])*T)
points(g::ChannelGrid, T, Nz, Nt)      = (g.y, (0:(Nz   - 1))/(Nz)*(2π/g.β),   (0:(Nt   - 1))/(Nt)*T)
