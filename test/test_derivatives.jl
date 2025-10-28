@testset "Field derivatives                     " begin
    # function definitions
    u_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    dudy_fun(y, z, t)   = -2*y*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    duds_fun(y, z, t)   = ((1 - y^2)*exp(cos(5.8*z))*cos(t))/(sin(t)^2 + 1)

    # construct grid
    Ny = 32; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # test values of derivatives
    u = FFT(PCField(g, u_fun, 2π))
    @test OpenChannelFlow.ddy!(  SCField(g), u) ≈ FFT(PCField(g, dudy_fun,   2π))
    @test OpenChannelFlow.d2dy2!(SCField(g), u) ≈ FFT(PCField(g, d2udy2_fun, 2π))
    @test OpenChannelFlow.ddz!(  SCField(g), u) ≈ FFT(PCField(g, dudz_fun,   2π))
    @test OpenChannelFlow.d2dz2!(SCField(g), u) ≈ FFT(PCField(g, d2udz2_fun, 2π))
    @test                 dds!(  SCField(g), u) ≈ FFT(PCField(g, duds_fun,   2π))

    # test time derivative of projected field
    M = 10
    Ψ = zeros(ComplexF64, Ny, M, (Nz >> 1) + 1, Nt)
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
        Ψ[:, :, nz, nt] .= qr(randn(ComplexF64, Ny, M)).Q[:, 1:M]
    end
    for m in 1:M
        OpenChannelFlow.apply_symmetry!(@view(Ψ[:, m, :, :]))
        Ψ[:, m, 1, 1] .= real.(Ψ[:, m, 1, 1])
    end
    a = project(FFT(VectorField(g, (u_fun,), 2π)), Ψ)
    @test dds!(similar(a), a) ≈ project(FFT(VectorField(g, (duds_fun,), 2π)), Ψ)

    # test allocation
    fun(dx, a, b) = @allocated dx(a, b)
    @test fun(OpenChannelFlow.ddy!,   SCField(g), u) == 0
    @test fun(OpenChannelFlow.d2dy2!, SCField(g), u) == 0
    @test fun(OpenChannelFlow.ddz!,   SCField(g), u) == 0
    @test fun(OpenChannelFlow.d2dz2!, SCField(g), u) == 0
    @test fun(                dds!,   SCField(g), u) == 0
    @test fun(                dds!,   similar(a), a) == 0
end
