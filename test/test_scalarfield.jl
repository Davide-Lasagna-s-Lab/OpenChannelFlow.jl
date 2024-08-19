@testset "Scalar Field                          " begin
    # fourier transforms
    # dot product
    # pointwise multiplication

    # define functions
    u_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    dudy_fun(y, z, t)   = -2*y*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    dudt_fun(y, z, t)   = ((1 - y^2)*exp(cos(5.8*z))*cos(t))/(sin(t)^2 + 1)

    # construct grid
    Ny = 32
    Nz = 33
    Nt = 51
    y = chebpts(Ny)
    D1 = chebdiff(Ny)
    D2 = chebddiff(Ny)
    w = rand(Float64, Ny)
    ω = 1.0
    β = 5.8
    grid = RPCFGrid(y, Nz, Nt, β, ω, D1, D2, w, flags=FFTW.ESTIMATE, dealias=false)
    _, z, t = points(grid)

    u = @test_nowarn RPCFField(grid)
    u.physical_field .= u_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    FFT!(u)

    dudt = RPCF.ddt!(RPCFField(grid), u)
    dudy = RPCF.ddy!(RPCFField(grid), u)
    d2udy2 = RPCF.d2dy2!(RPCFField(grid), u)
    dudz = RPCF.ddz!(RPCFField(grid), u)
    d2udz2 = RPCF.d2dz2!(RPCFField(grid), u)

    IFFT!(dudt)
    IFFT!(dudy)
    IFFT!(d2udy2)
    IFFT!(dudz)
    IFFT!(d2udz2)

    @test dudt.physical_field ≈ dudt_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test dudy.physical_field ≈ dudy_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test d2udy2.physical_field ≈ d2udy2_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test dudz.physical_field ≈ dudz_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test d2udz2.physical_field ≈ d2udz2_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
end