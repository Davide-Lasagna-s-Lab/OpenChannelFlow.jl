module RPCF

using ReSolverInterface

export points

struct Grid <: AbstractGrid{Float64, 3} end
ReSolverInterface.points(::Grid) = 5
ReSolverInterface.volume(::Grid) = 1

end
