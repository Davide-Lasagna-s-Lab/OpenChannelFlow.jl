# Utility functions to convert a set of DNS to a scalar field

function data2ScalarField(data, grid::RPCFGrid{S}) where {S}
    Ny, Nz, Nt = size(data)
    u = VectorField(RPCFField, grid)
    A = [Array{Float64, 3}(undef, Ny, Nz, Nt) for _ in 1:3]
    for n in 1:3, (i, snap) in enumerate(data)
        A[n][:, :, i] .= snap[n]
    end
    B = [rfft(A[i], [2, 3])./(Nz*Nt) for i in 1:3]
    for n in 1:3, nz in 1:((minimum([S[2], Nz]) >> 1) + 1)
        u[n][:, nz, 1] .= B[n][:, nz, 1]
    end
    for n in 1:3, nz in 1:((minimum([S[2], Nz]) >> 1) + 1), nt in 2:((minimum([S[3], Nt]) >> 1) + 1)
        u[n][:, nz, nt] .= B[n][:, nz, nt]
        u[n][:, nz, end-nt+2] .= B[n][:, nz, end-nt+2]
    end
    return u
end
data2Coefficients(data, grid::RPCFGrid, modes) = expand!(ReSolverInterface.ProjectedField(grid, modes), data2SpectralField(data, grid))
