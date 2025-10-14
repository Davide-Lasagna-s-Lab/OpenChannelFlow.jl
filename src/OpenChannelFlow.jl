module OpenChannelFlow

using FFTW

export ChannelGrid, points, growto
export ModeNumber
export SCField, grid
export PCField
export VectorField
export FFT, IFFT
# export ProjectedField, expand!, project!

include("grid.jl")
include("modenumber.jl")
include("scfield.jl")
include("pcfield.jl")
include("vectorfield.jl")
# include("broadcasting.jl")
include("fft.jl")

# TODO: try to make this compatible with the flows interface as well as the ReSolver one?

end
