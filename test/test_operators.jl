@testset "Implicit operator                     " begin
    # function definitions
    Re = rand()*50
    u_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    v_fun(y, z, t)      = cos(π*y/2)*exp(sin(5.8*z))*cos(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    d2vdy2_fun(y, z, t) = -(π^2/4)*cos(π*y/2)*exp(sin(5.8*z))*cos(sin(t))
    d2vdz2_fun(y, z, t) = (5.8^2)*cos(π*y/2)*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(sin(t))
    Δu_fun(y, z, t)     = (d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t))/Re
    Δv_fun(y, z, t)     = (d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t))/Re

    # construct grid
    Ny = 32; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # test laplacian calculation
    op = OpenChannelFlow.ImplicitOperator(Re)
    u = FFT(VectorField(g, (u_fun, v_fun,), 2π))
    @test op(0.0, u, similar(u)) ≈ FFT(VectorField(g, (Δu_fun, Δv_fun), 2π))
end

@testset "Explicit operator                     " begin
    # function definitions
    u_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    v_fun(y, z, t)      = cos(π*y/2)^2*exp(sin(5.8*z))*cos(sin(t))
    w_fun(y, z, t)      = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*cos(t)^2
    dudy_fun(y, z, t)   = -2*y*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    dvdy_fun(y, z, t)   = -(π/2)*sin(π*y)*exp(sin(5.8*z))*cos(sin(t))
    dvdz_fun(y, z, t)   = 5.8*cos(π*y/2)^2*cos(5.8*z)*exp(sin(5.8*z))*cos(sin(t))
    dwdy_fun(y, z, t)   = -(π*sin(π*y)*(1 - y^2) + 2*y*cos(π*y))*exp(sin(5.8*z))*cos(t)^2
    dwdz_fun(y, z, t)   = 5.8*cos(π*y)*(1 - y^2)*cos(5.8*z)*exp(sin(5.8*z))*cos(t)^2
    u_out_fun(y, z, t)  = -v_fun(y, z, t)*dudy_fun(y, z, t) - w_fun(y, z, t)*dudz_fun(y, z, t) + Ro*v_fun(y, z, t)
    v_out_fun(y, z, t)  = -v_fun(y, z, t)*dvdy_fun(y, z, t) - w_fun(y, z, t)*dvdz_fun(y, z, t) - Ro*u_fun(y, z, t)
    w_out_fun(y, z, t)  = -v_fun(y, z, t)*dwdy_fun(y, z, t) - w_fun(y, z, t)*dwdz_fun(y, z, t)

    # construct grid
    Ny = 32; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # do something
    Ro = rand()
    op = OpenChannelFlow.ExplicitOperator(g, Ro, Float64, FFTW.ESTIMATE)
    u = FFT(VectorField(g, (u_fun, v_fun, w_fun), 2π))
    out = similar(u)
    op(0.0, u, out, add=false)
    yes = FFT(VectorField(g, (u_out_fun, v_out_fun, w_out_fun), 2π))
    @test op(0.0, u, out, add=false) ≈ FFT(VectorField(g, (u_out_fun, v_out_fun, w_out_fun), 2π))
    @test op(0.0, u, out, add=true)  ≈ FFT(VectorField(g, (u_out_fun, v_out_fun, w_out_fun), 2π)).*2
end
