@testset "Field symmetry shifts                 " begin
    # define functions
    sz = 2π*rand(); st = rand()
    u_fun       = ((y, z, t)->y + (1 - y^2)*cos(z)*cos(t),
                   (y, z, t)->-(π/2)*cos(π*y/2)^2*sin(z)*sin(t),
                   (y, z, t)->(π/2)*sin(π*y)*cos(z)*sin(t))
    u_shift_fun = ((y, z, t)->y + (1 - y^2)*cos(z + sz)*cos(t + st*2π),
                   (y, z, t)->-(π/2)*cos(π*y/2)^2*sin(z + sz)*sin(t + st*2π),
                   (y, z, t)->(π/2)*sin(π*y)*cos(z + sz)*sin(t + st*2π))

    # construct grid
    Ny = 16; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # generate modes
    Random.seed!(0)
    M = 10
    Ψ = zeros(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
        Ψ[:, :, nz, nt] .= qr(randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
    end
    for m in 1:M
        OpenChannelFlow.apply_symmetry!(@view(Ψ[:, m, :, :]))
        Ψ[:, m, 1, 1] .= real.(Ψ[:, m, 1, 1])
    end

    # test shifts
    u       = FFT(VectorField(g, u_fun,       2π))
    u_shift = FFT(VectorField(g, u_shift_fun, 2π))
    @test shift!(     u,  (0,  0))  === u
    @test shift!(copy(u), (sz, st)) ≈   u_shift atol=1e-12
    a       = project(u,       Ψ)
    a_shift = project(u_shift, Ψ)
    @test shift!(     a,  (0,  0),  1.0) === a
    @test shift!(copy(a), (sz, st), 1.0) ≈   a_shift atol=1e-12
end
