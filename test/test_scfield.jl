@testset "Spectral channel field                " begin
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
    A = randn(ComplexF64, Ny, (Nz >> 1) + 1, Nt)
    T = rand([Float64, Float32, Float16])

    # test constructor
    @test_nowarn SCField(g, A)
    @test_nowarn SCField(g)
    @test_nowarn SCField(g, T)

    # test interfaces
    @test eltype(SCField(g, T)) == Complex{T}
    @test size(SCField(g)) == (Ny, (Nz >> 1) + 1, Nt)
    @test similar(SCField(g, A)) isa SCField{typeof(g), Float64}
    @test similar(SCField(g, A), Complex{T}) isa SCField{typeof(g), T}
    @test copy(SCField(g, A)) == SCField(g, A)
    @test zero(SCField(g, A)) == SCField(g, zero(A))
    @test abs(SCField(g, A)) == SCField(g, ComplexF64.(abs.(A)))

    # test array indexing
    u = SCField(g, A)
    for ny in 1:Ny, nz in 1:(Nz >> 1) + 1, nt in 1:Nt
        @test u[ny, nz, nt] == A[ny, nz, nt]
    end

    # test mode number indexing
    for ny in 1:Ny, nt in 0:(Nz >> 1)
        @test u[ny, ModeNumber(0,  nt)] ==      A[ny, 1, nt+1]
        @test u[ny, ModeNumber(0, -nt)] == conj(A[ny, 1, nt+1])
    end
    for ny in 1:Ny, nz in 1:32
        @test u[ny, ModeNumber( nz,  0)] ==      A[ny, nz+1, 1]
        @test u[ny, ModeNumber(-nz,  0)] == conj(A[ny, nz+1, 1])
    end
    for ny in 1:Ny, nz in 1:32, nt in 1:32
        @test u[ny, ModeNumber( nz,  nt)] ==      A[ny, nz+1,    nt+1]
        @test u[ny, ModeNumber( nz, -nt)] ==      A[ny, nz+1, Nt-nt+1]
        @test u[ny, ModeNumber(-nz,  nt)] == conj(A[ny, nz+1, Nt-nt+1])
        @test u[ny, ModeNumber(-nz, -nt)] == conj(A[ny, nz+1,    nt+1])
    end
end
