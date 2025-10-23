using Test

using FFTW, Random, LinearAlgebra

using ChebUtils

using OpenChannelFlow

include("test_grid.jl")
include("test_modenumber.jl")
include("test_scfield.jl")
include("test_pcfield.jl")
include("test_vectorfield.jl")
include("test_projectedfield.jl")
include("test_broadcasting.jl")
include("test_fft.jl")
include("test_derivatives.jl")
include("test_shifts.jl")
# include("test_norms.jl")
