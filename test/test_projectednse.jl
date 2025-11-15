@testset "Projected NS Operators                " begin
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
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), m in 1:M
        Ψ[     1, m, nz, nt] = 0.0
        Ψ[  Ny  , m, nz, nt] = 0.0
        Ψ[  Ny+1, m, nz, nt] = 0.0
        Ψ[2*Ny  , m, nz, nt] = 0.0
        Ψ[2*Ny+1, m, nz, nt] = 0.0
        Ψ[3*Ny  , m, nz, nt] = 0.0
    end

    # generate fields
    a = ProjectedField(g, Ψ)
    b = ProjectedField(g, Ψ)
    a .= randn(ComplexF64, M, (Nz >> 1) + 1, Nt)
    b .= randn(ComplexF64, M, (Nz >> 1) + 1, Nt)
    a[:, 1, 1] .= real.(a[:, 1, 1])
    b[:, 1, 1] .= real.(b[:, 1, 1])
    OpenChannelFlow.apply_symmetry!(a)
    OpenChannelFlow.apply_symmetry!(b)
    u = expand(a)
    u[1][:, 1, 1] .+= g.y
    v = expand(b)

    # test projected equations
    Re = rand()*50
    Ro = rand()
    op_nl = CouettePrimitiveNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    op_ln = CouettePrimitiveLNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    op_pr = ProjectedNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    @test op_pr(similar(a), a)    ≈ project(op_nl(0, u, similar(u)), Ψ)
    @test op_pr(similar(a), a, b) ≈ project(op_ln(0, u, v, similar(u), true), Ψ)
end
