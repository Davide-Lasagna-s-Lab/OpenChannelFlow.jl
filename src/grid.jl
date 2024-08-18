# Implementation of the RPCF grid

struct RPCFGrid{S, DM<:AbstractMatrix, DEALIAS, PLAN, IPLAN} <: AbstractGrid{Float64, 3}
    y::Vector{Float64}
    Nz::Int
    Nt::Int
    domain::Vector{Float64}
    Dy::DM
    Dy2::DM
    ws::Vector{Float64}
    plans::FFTPlans{DEALIAS, PLAN, IPLAN}

    function RPCFGrid(y, Nz, Nt, β, ω, Dy::DM, Dy2::DM, ws; dealias::Bool=true, pad::Float64=3/2, flags::UInt32=FFTW.EXHAUSTIVE, timelimit::Real=FFTW.NO_TIMELIMIT) where {DM}
        plans = FFTPlans(length(y), Nz, Nt, dealias=dealias, pad=pad, flags=flags, timelimit=timelimit)
        new{(length(y), Nz, Nt), DM, dealias, get_plan_types(plans)...}(y, Nz, Nt, [β, ω], Dy, Dy2, ws, plans)
    end
end

# getter-setter methods
function Base.getproperty(g::RPCFGrid, name::Symbol)
    if name === :L
        return 2π/g.domain[1]
    elseif name === :β
        return g.domain[1]
    elseif name === :T
        return 2π/g.domain[2]
    elseif name === :ω
        return g.domain[2]
    else
        return getfield(g, name)
    end
end

Base.size(::RPCFGrid{S}) where {S} = S

# interface methods
ReSolverInterface.points(g::RPCFGrid{S}) where {S} = (g.y, ntuple(i -> (0:(S[i + 1] - 1))/(S[i + 1])*(2π/g.domain[i]), 2)...)
ReSolverInterface.volume(g::RPCFGrid) = 2*g.L*g.T
