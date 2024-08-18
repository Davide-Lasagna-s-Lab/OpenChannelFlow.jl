module RPCF

using FFTW

using ReSolverInterface

export RPCFGrid, points, volume

include("fft.jl")
include("grid.jl")
include("scalarfield.jl")

end
