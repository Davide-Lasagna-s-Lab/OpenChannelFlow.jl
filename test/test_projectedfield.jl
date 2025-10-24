@testset "Projected channel field               " begin
    # construct grid
    Ny = 16; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    ones(Ny))

    # generate modes
    M = 10
    Ψ = zeros(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
        Ψ[:, :, nz, nt] .= qr(randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
    end

    # test constructor
    @test ProjectedField(g, Ψ) isa ProjectedField{typeof(g), 10, Float64, Array{ComplexF64, 4}}
    @test ProjectedField(g, zeros(ComplexF64, M, (Nz >> 1) + 1, Nt), Ψ) isa ProjectedField{typeof(g), 10, Float64, Array{ComplexF64, 4}}
    @test_throws ArgumentError ProjectedField(g, zeros(ComplexF64, M+1, (Nz >> 1) + 1, Nt), Ψ)

    # test channel integration
    u = ComplexF64[(y^2)*cos(π*y/2) for y in g.y]
    v = ComplexF64[exp(-5*(y^2)) for y in g.y]
    @test OpenChannelFlow.channel_int(u, chebws(Ny), v, Ny) ≈ 0.0530025 rtol=1e-5

    # test project and expand
    a = ProjectedField(g, Ψ)
    parent(a) .= randn(ComplexF64, M, (Nz >> 1) + 1, Nt)
    u = VectorField(g)
    for nz in 1:((Nz >> 1) + 1), nt in 1:Nt
        u[1][:, nz, nt] .= Ψ[     1:Ny,   :, nz, nt]*a[:, nz, nt]
        u[2][:, nz, nt] .= Ψ[  Ny+1:2*Ny, :, nz, nt]*a[:, nz, nt]
        u[3][:, nz, nt] .= Ψ[2*Ny+1:3*Ny, :, nz, nt]*a[:, nz, nt]
    end
    @test project(u, Ψ) ≈ a
    @test expand(a) ≈ u
end
