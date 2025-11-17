@testset "Norm weighting                        " begin
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

    # construct fields
    a = project(FFT(VectorField(g, (f1,), 2π)), Ψ)
    b = project(FFT(VectorField(g, (f2,), 2π)), Ψ)

    # test weighting operation
    A = FarazmandWeight(2π, 2π)
    c = copy(a)
    for nz in 1:(Nz >> 1) + 1, m in 1:M
        c[m, nz, 1] /= 1 + 4π^2*(nz - 1)^2
    end
    for nt in 2:(Nt >> 1) + 1, nz in 1:(Nz >> 1) + 1, m in 1:M
        c[m, nz, nt]       /= 1 + 4π^2*(nz - 1)^2 + 4π^2*(nt - 1)^2
        c[m, nz, end-nt+2] /= 1 + 4π^2*(nz - 1)^2 + 4π^2*(nt - 1)^2
    end
    @test mul!(copy(a), A) ≈ c
    @test dot(a, A, b) == dot(a, mul!(copy(b), A))
end
