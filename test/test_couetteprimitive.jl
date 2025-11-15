# @testset "Couette primitive NSE                 " begin
#     # define functions
#     u_fun(y, z, t)      = y + (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
#     dudy_fun(y, z, t)   = 1 - 2*y*exp(cos(5.8*z))*atan(sin(t))
#     d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
#     dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
#     d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
#     v_fun(y, z, t)      = cos(π*y/2)^2*exp(sin(5.8*z))*cos(sin(t))
#     dvdy_fun(y, z, t)   = -(π/2)*sin(π*y)*exp(sin(5.8*z))*cos(sin(t))
#     d2vdy2_fun(y, z, t) = -(π^2/2)*cos(π*y)*exp(sin(5.8*z))*cos(sin(t))
#     dvdz_fun(y, z, t)   = 5.8*cos(π*y/2)^2*cos(5.8*z)*exp(sin(5.8*z))*cos(sin(t))
#     d2vdz2_fun(y, z, t) = (5.8^2)*cos(π*y/2)^2*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(sin(t))
#     w_fun(y, z, t)      = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*cos(t)^2
#     dwdy_fun(y, z, t)   = -(π*sin(π*y)*(1 - y^2) + 2*y*cos(π*y))*exp(sin(5.8*z))*cos(t)^2
#     d2wdy2_fun(y, z, t) = -(π^2*cos(π*y)*(1 - y^2) - 4π*y*sin(π*y) + 2*cos(π*y))*exp(sin(5.8*z))*cos(t)^2
#     dwdz_fun(y, z, t)   = 5.8*cos(π*y)*(1 - y^2)*cos(5.8*z)*exp(sin(5.8*z))*cos(t)^2
#     d2wdz2_fun(y, z, t) = (5.8^2)*cos(π*y)*(1 - y^2)*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(t)^2
#     u_out_fun(y, z, t)  = (d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t))/Re - v_fun(y, z, t)*dudy_fun(y, z, t) - w_fun(y, z, t)*dudz_fun(y, z, t) + Ro*v_fun(y, z, t)
#     v_out_fun(y, z, t)  = (d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t))/Re - v_fun(y, z, t)*dvdy_fun(y, z, t) - w_fun(y, z, t)*dvdz_fun(y, z, t) - Ro*u_fun(y, z, t)
#     w_out_fun(y, z, t)  = (d2wdy2_fun(y, z, t) + d2wdz2_fun(y, z, t))/Re - v_fun(y, z, t)*dwdy_fun(y, z, t) - w_fun(y, z, t)*dwdz_fun(y, z, t)

#     # construct grid
#     Ny = 32; Nz = 33; Nt = 51
#     g = ChannelGrid(chebpts(Ny), Nz, Nt,
#                     5.8,
#                     chebdiff(Ny),
#                     chebddiff(Ny),
#                     chebws(Ny))

#     # test nonlinear operator
#     Re = rand()*50
#     Ro = rand()
#     op = CouettePrimitiveNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
#     u = FFT(VectorField(g, (u_fun, v_fun, w_fun), 2π))
#     exact = FFT(VectorField(g, (u_out_fun, v_out_fun, w_out_fun), 2π))
#     @test op(0.0, u, similar(u)) ≈ exact
# end

@testset "Couette primitive linearised NSE      " begin
    # define functions
    ux_fun(y, z, t) = y + (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    uy_fun(y, z, t) = cos(π*y/2)^4*sin(5.8*z)*cos(sin(t))
    uz_fun(y, z, t) = (1 - y^2)*cos(5.8*z)*cos(cos(t))
    vx_fun(y, z, t) = cos(π*y/2)^2*exp(sin(5.8*z))*cos(sin(t))
    vy_fun(y, z, t) = (1 - y^2)*exp(cos(5.8*z))*atan(cos(t))
    vz_fun(y, z, t) = sin(π*y)*sin(z)^2*cos(t)
    wx_fun(y, z, t) = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*cos(t)^2
    wy_fun(y, z, t) = cos(π*y/2)*cos(5.8*z)*sin(t)^2
    wz_fun(y, z, t) = cos(π*y/2)^4*sin(5.8*z)*atan(cos(t))
    duydy_fun(y, z, t)    = -2π*sin(π*y/2)*cos(π*y/2)^3*sin(5.8*z)*cos(sin(t))
    duzdz_fun(y, z, t)    = -5.8*(1 - y^2)*sin(5.8*z)*cos(cos(t))
    div_u_wx_fun(y, z, t) = (duydy_fun(y, z, t) + duzdz_fun(y, z, t))*wx_fun(y, z, t)
    div_u_wy_fun(y, z, t) = (duydy_fun(y, z, t) + duzdz_fun(y, z, t))*wy_fun(y, z, t)
    div_u_wz_fun(y, z, t) = (duydy_fun(y, z, t) + duzdz_fun(y, z, t))*wz_fun(y, z, t)

    # construct grid
    Ny = 32; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # define fields
    u       = FFT(VectorField(g, (ux_fun, uy_fun, uz_fun), 2π))
    v       = FFT(VectorField(g, (vx_fun, vy_fun, vz_fun), 2π))
    w       = FFT(VectorField(g, (wx_fun, wy_fun, wz_fun), 2π))
    div_u_w = FFT(VectorField(g, (div_u_wx_fun, div_u_wy_fun, div_u_wz_fun), 2π))

    # test perturbed nonlinear equations approximates linearised equations
    Re = rand()*50
    Ro = rand()
    op_nl = CouettePrimitiveNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    op_ln = CouettePrimitiveLNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    a = op_nl(0.0, u .+ 1e-6.*v, similar(u)) - op_nl(0.0, u, similar(u))
    b = op_ln(0.0, u, 1e-6.*v, similar(u), false)
    @test norm(a - b) < 1e-11

    # test adjoint identity
    @test dot(op_ln(0.0, u, v, similar(u), false), w) ≈ dot(v, op_ln(0.0, u, w, similar(u), true)) + dot(v, div_u_w)
    # extra term in adjoint operator is required for field `u` that isn't divergence-free
end
