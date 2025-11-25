module OpenChannelFlow

using FFTW, LinearAlgebra

export ChannelGrid, points, growto, get_fields
export ModeNumber
export SCField, grid
export PCField
export VectorField
export ProjectedField, modes, project!, project, expand!, expand
export FFT, IFFT
export dds!
export shift!
export dot, norm, normdiff, minnormdiff
export FarazmandWeight
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE
export ProjectedNSE

include("grid.jl")
include("modenumber.jl")
include("scfield.jl")
include("pcfield.jl")
include("vectorfield.jl")
include("fft.jl")
include("projectedfield.jl")
include("broadcasting.jl")
include("derivatives.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("cartesianprimitive.jl")
include("projectednse.jl")

end
