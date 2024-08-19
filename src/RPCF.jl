module RPCF

using FFTW

using ReSolverInterface

export RPCFGrid, points, volume
export RPCFField, FFT!, IFFT!, dot, norm, mult!

include("fft.jl")
include("grid.jl")
include("scalarfield.jl")
include("vectorcalculus.jl")

end
