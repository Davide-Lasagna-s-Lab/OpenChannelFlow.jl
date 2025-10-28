@testset "Field norms                           " begin
    # function definitions
    f1(y, z, t) =          (1 - y^2)*exp(cos(z))*cos(sin(t))
    f2(y, z, t) = cos(π*y)*(1 - y^2)*exp(sin(z))*cos(t)^2

    # construct grid
    Ny = 32; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # generate modes
    M = Ny
    Ψ = zeros(ComplexF64, Ny, M, (Nz >> 1) + 1, Nt)
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1)
        Ψ[:, :, nz, nt] .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M]
    end
    for nt in 2:Nt
        Ψ[:, :, 1,     nt]   .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, end-nt+2] .= conj.(Ψ[:, :, 1, nt])
    end
    Ψ[:, :, 1, 1] .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(Float64, Ny, M)).Q[:, 1:M]

    # test norms of channel fields
    @test dot(FFT(PCField(g, f1, 2π)), FFT(PCField(g, f2, 2π))) ≈ 0.169796 rtol=1e-5
    @test norm(FFT(PCField(g, f1, 2π)))^2                       ≈ 0.743990 rtol=1e-5
    @test norm(FFT(VectorField(g, (f1, f2), 2π)))^2             ≈ 0.965370 rtol=1e-5
    @test norm(project(FFT(VectorField(g, (f1,), 2π)), Ψ))      ≈ norm(FFT(VectorField(g, (f1,), 2π)))

    # test norm difference methods
    @test normdiff(FFT(VectorField(g, (f1,), 2π)), FFT(VectorField(g, (f2,), 2π)))^2 ≈ 0.625777 rtol=1e-5
    @test normdiff(project(FFT(VectorField(g, (f1,), 2π)), Ψ), project(FFT(VectorField(g, (f2,), 2π)), Ψ))^2 ≈ 0.625777 rtol=1e-5
    mindiff, s_mins = minnormdiff(FFT(PCField(g, f1, 2π)), FFT(PCField(g, (y, z, t)->f1(y, z-π, t+π/2), 2π)), (4, 4))
    @test all(s_mins .≈ (π, 0.25))
end
