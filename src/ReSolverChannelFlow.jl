module ReSolverChannelFlow

using FFTW, LinearAlgebra

using NSEBase

export ChannelGrid, points, growto, get_fields
export ModeNumber
export FTField, grid
export Field
export VectorField
export ProjectedField, modes, project!, project, expand!, expand
export FFT, IFFT, FFTPlans
export dds!, ddx1!, ddx2!, ddx3!, laplacian!
export shift!
export dot, norm, normdiff, minnormdiff
export FarazmandWeight
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE
export ProjectedNSE

include("grid.jl")
include("modenumber.jl")
include("ftfield.jl")
include("field.jl")
include("derivatives.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("cartesianprimitive.jl")

end
