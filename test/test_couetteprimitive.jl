@testset "Couette primitive NSE                 " begin
    # define functions
    u_fun(y, z, t)      = y + (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    dudy_fun(y, z, t)   = 1 - 2*y*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    v_fun(y, z, t)      = cos(π*y/2)^2*exp(sin(5.8*z))*cos(sin(t))
    dvdy_fun(y, z, t)   = -(π/2)*sin(π*y)*exp(sin(5.8*z))*cos(sin(t))
    d2vdy2_fun(y, z, t) = -(π^2/2)*cos(π*y)*exp(sin(5.8*z))*cos(sin(t))
    dvdz_fun(y, z, t)   = 5.8*cos(π*y/2)^2*cos(5.8*z)*exp(sin(5.8*z))*cos(sin(t))
    d2vdz2_fun(y, z, t) = (5.8^2)*cos(π*y/2)^2*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(sin(t))
    w_fun(y, z, t)      = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*cos(t)^2
    dwdy_fun(y, z, t)   = -(π*sin(π*y)*(1 - y^2) + 2*y*cos(π*y))*exp(sin(5.8*z))*cos(t)^2
    d2wdy2_fun(y, z, t) = -(π^2*cos(π*y)*(1 - y^2) - 4π*y*sin(π*y) + 2*cos(π*y))*exp(sin(5.8*z))*cos(t)^2
    dwdz_fun(y, z, t)   = 5.8*cos(π*y)*(1 - y^2)*cos(5.8*z)*exp(sin(5.8*z))*cos(t)^2
    d2wdz2_fun(y, z, t) = (5.8^2)*cos(π*y)*(1 - y^2)*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(t)^2
    u_out_fun(y, z, t)  = (d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t))/Re - v_fun(y, z, t)*dudy_fun(y, z, t) - w_fun(y, z, t)*dudz_fun(y, z, t) + Ro*v_fun(y, z, t)
    v_out_fun(y, z, t)  = (d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t))/Re - v_fun(y, z, t)*dvdy_fun(y, z, t) - w_fun(y, z, t)*dvdz_fun(y, z, t) - Ro*u_fun(y, z, t)
    w_out_fun(y, z, t)  = (d2wdy2_fun(y, z, t) + d2wdz2_fun(y, z, t))/Re - v_fun(y, z, t)*dwdy_fun(y, z, t) - w_fun(y, z, t)*dwdz_fun(y, z, t)

    # construct grid
    Ny = 32; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # construct modes
    M = 3*Ny
    Ψ = zeros(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1)
        Ψ[:, :, nz, nt] .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
    end
    for nt in 2:Nt
        Ψ[:, :, 1,     nt]   .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, end-nt+2] .= conj.(Ψ[:, :, 1, nt])
    end
    Ψ[:, :, 1, 1] .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(Float64, 3*Ny, M)).Q[:, 1:M]

    # test nonlinear operator
    Random.seed!(0)
    Re = rand()*50
    Ro = rand()
    op = CouettePrimitiveNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    u = FFT(VectorField(g, (u_fun, v_fun, w_fun), 2π))
    exact = FFT(VectorField(g, (u_out_fun, v_out_fun, w_out_fun), 2π))
    @test op(0.0, u, similar(u)) ≈ exact
    u[1][:, 1, 1] .-= g.y
    a = project(u, Ψ)
    @test op(similar(a), a) ≈ project(exact, Ψ)
end

@testset "Couette primitive linearised NSE      " begin

end
