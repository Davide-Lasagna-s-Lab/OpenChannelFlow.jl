module OpenChannelFlow

using FFTW, LinearAlgebra

export ChannelGrid, points, growto
export ModeNumber
export SCField, grid
export PCField
export VectorField
export FFT, IFFT
export dds!
# export ProjectedField, expand!, project!

include("grid.jl")
include("modenumber.jl")
include("scfield.jl")
include("pcfield.jl")
include("vectorfield.jl")
include("broadcasting.jl")
include("fft.jl")
include("derivatives.jl")
# include("projectedfield.jl")
# include("norms.jl")
# include("nonlinear.jl")
# include("linearised.jl")

# TODO: try to make this compatible with the flows interface as well as the ReSolver one?

end
