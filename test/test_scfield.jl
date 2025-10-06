@testset "Spectral channel field" begin
    # construct grid
    Ny = 32; Nz = 65; Nt = 65
    g = ChannelGrid(chebpts(Ny),
                    Nz, Nt,
                    5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    flags=FFTW.ESTIMATE,
                    dealias=false)

    # some random variables
    A = randn(Ny, (Nz >> 1) + 1, Nt)
    T = rand([Int64, Int32, Float32, Float16])

    # test constructor
    @test_nowarn SCField(g, A)
    @test_nowarn SCField(g)
    @test_nowarn SCField(g, T)

    # test interfaces
    @test eltype(SCField(g, T)) == Complex{T}
    @test size(SCField(g)) == (Ny, (Nz >> 1) + 1, Nt)
    @test similar(SCField(g, A)) isa SCField{typeof(g), Float64}
    @test similar(SCField(g, A), T) isa SCField{typeof(g), T}
    @test copy(SCField(g, A)) == SCField(g, A)
    @test zero(SCField(g, A)) == SCField(g, zero(A))
    @test abs(SCField(g, A)) == SCField(g, abs.(A))

    # test indexing
    # u = SCField(g, A)
    
end
