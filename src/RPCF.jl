module RPCF

using FFTW

using ReSolverInterface

export RPCFGrid, points, volume
export RPCFField, FFT!, IFFT!, dot, norm, mult!
export VectorField, grad!, divergence!, laplacian!, convection!, convection2!
export Objective

include("fft.jl")
include("grid.jl")
include("scalarfield.jl")
include("vectorcalculus.jl")
include("operators.jl")
include("dns2field.jl")

end
