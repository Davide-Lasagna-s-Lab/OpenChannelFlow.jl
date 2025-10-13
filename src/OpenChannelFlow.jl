module OpenChannelFlow

using FFTW

export ChannelGrid, points, growto!
export ModeNumber
export SCField, grid
export PCField
export FFT, IFFT
# export VectorField, grad!, divergence!, laplacian!, convection!, convection2!
# export ProjectedField, expand!, project!

include("grid.jl")
include("modenumber.jl")
include("scfield.jl")
include("pcfield.jl")
# include("vectorfield.jl")
# include("broadcasting.jl")
include("fft.jl")
# include("vectorcalculus.jl")
# include("operators.jl")
# include("dns2field.jl")

# TODO: try to make this compatible with the flows interface as well as the ReSolver one?

end
