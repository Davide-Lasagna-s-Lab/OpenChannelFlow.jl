@testset "Vector Calculus                       " begin
    # define functions
    u_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    v_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*cos(sin(ω*t))
    w_fun(y, z, t)      = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*(cos(ω*t)^2)
    dudx_fun(y, z, t)   = 0.0
    dudy_fun(y, z, t)   = -2*y*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    dvdy_fun(y, z, t)   = -2*y*exp(cos(5.8*z))*cos(sin(ω*t))
    dvdz_fun(y, z, t)   = -5.8*(1 - y^2)*(sin(5.8*z)*exp(cos(5.8*z)))*cos(sin(ω*t))
    dwdy_fun(y, z, t)   = -(π*sin(π*y)*(1 - y^2) + cos(π*y)*2*y)*exp(sin(5.8*z))*(cos(ω*t)^2)
    dwdz_fun(y, z, t)   = 5.8*cos(π*y)*(1 - y^2)*cos(5.8*z)*exp(sin(5.8*z))*(cos(ω*t)^2)
    div_u_fun(y, z, t)  = dvdy_fun(y, z, t) + dwdz_fun(y, z, t)
    Δu_fun(y, z, t)     = d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t)
    u∇u_fun(y, z, t)    = v_fun(y, z, t)*dudy_fun(y, z, t) + w_fun(y, z, t)*dudz_fun(y, z, t)
    u∇v_fun(y, z, t)    = v_fun(y, z, t)*dvdy_fun(y, z, t) + w_fun(y, z, t)*dvdz_fun(y, z, t)
    u∇w_fun(y, z, t)    = v_fun(y, z, t)*dwdy_fun(y, z, t) + w_fun(y, z, t)*dwdz_fun(y, z, t)
    ∇uu_fun(y, z, t)    = 0.0
    ∇uv_fun(y, z, t)    = u_fun(y, z, t)*dudy_fun(y, z, t) + v_fun(y, z, t)*dvdy_fun(y, z, t) + w_fun(y, z, t)*dwdy_fun(y, z, t)
    ∇uw_fun(y, z, t)    = u_fun(y, z, t)*dudz_fun(y, z, t) + v_fun(y, z, t)*dvdz_fun(y, z, t) + w_fun(y, z, t)*dwdz_fun(y, z, t)

    # construct grid
    Ny = 32
    Nz = 33
    Nt = 51
    y = chebpts(Ny)
    D1 = chebdiff(Ny)
    D2 = chebddiff(Ny)
    w = chebws(Ny)
    ω = 1.0
    β = 5.8
    grid = RPCFGrid(y, Nz, Nt, β, ω, D1, D2, w, flags=FFTW.ESTIMATE, dealias=true)
    _, z, t = points(grid)

    u = VectorField(RPCFField, grid)
    u[1].physical_field .= u_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    u[2].physical_field .= v_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    u[3].physical_field .= w_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    FFT!(u)

    ∇u    = grad!(VectorField(RPCFField, grid), u[1])
    div_u = divergence!(RPCFField(grid), u)
    Δu    = laplacian!(RPCFField(grid), u[1])
    # FIXME: the fallback method does not work?
    u∇u   = convection!(VectorField(RPCFField, grid), u, u)
    ∇uu   = convection2!(VectorField(RPCFField, grid), u, u)
    IFFT!(∇u)
    IFFT!(div_u)
    IFFT!(Δu)
    IFFT!(u∇u)
    IFFT!(∇uu)

    @test ∇u[1].physical_field  == dudx_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test ∇u[2].physical_field  ≈  dudy_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test ∇u[3].physical_field  ≈  dudz_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test div_u.physical_field  ≈  div_u_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test Δu.physical_field     ≈  Δu_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test u∇u[1].physical_field ≈  u∇u_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test u∇u[2].physical_field ≈  u∇v_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test u∇u[3].physical_field ≈  u∇w_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test ∇uu[1].physical_field ≈  ∇uu_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test ∇uu[2].physical_field ≈  ∇uv_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
    @test ∇uu[3].physical_field ≈  ∇uw_fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
end