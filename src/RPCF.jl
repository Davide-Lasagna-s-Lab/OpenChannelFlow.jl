module RPCF

using ReSolverInterface

export RPCFGrid, points, volume

include("fft.jl")
include("grid.jl")
include("scalarfield.jl")

end
