@testset "Spectral channel field                " begin
    # construct grid
    Ny = 16; Nx=15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # some random variables
    A = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)
    T = rand([Float64, Float32, Float16])

    # test constructor
    @test_nowarn SCField(g, A)
    @test_nowarn SCField(g)
    @test_nowarn SCField(g, T)

    # test interfaces
    @test eltype(SCField(g, T)) == Complex{T}
    @test size(SCField(g)) == (Ny, (Nx >> 1) + 1, Nz, Nt)
    @test similar(SCField(g, A)) isa SCField{typeof(g), Float64}
    @test similar(SCField(g, A), Complex{T}) isa SCField{typeof(g), T}
    @test copy(SCField(g, A)) == SCField(g, A)
    @test zero(SCField(g, A)) == SCField(g, zero(A))
    @test abs(SCField(g, A)) == SCField(g, ComplexF64.(abs.(A)))

    # test array indexing
    u = SCField(g, A)
    for ny in 1:Ny, nx in 1:(Nx >> 1) + 1, nz in 1:Nz, nt in 1:Nt
        @test u[ny, nx, nz, nt] == A[ny, nx, nz, nt]
    end

    # test mode number indexing
    for ny in 1:Ny
        @test u[ny, ModeNumber(0, 0, 0)] == A[ny, 1, 1, 1]
    end
    for ny in 1:Ny, nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber(0, 0,  nt)] == A[ny, 1, 1,    nt+1]
        @test u[ny, ModeNumber(0, 0, -nt)] == A[ny, 1, 1, Nt-nt+1]
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1)
        @test u[ny, ModeNumber(0,  nz, 0)] ==      A[ny, 1, nz+1, 1]
        @test u[ny, ModeNumber(0, -nz, 0)] == conj(A[ny, 1, nz+1, 1])
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1), nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber(0,  nz,  nt)] ==      A[ny, 1, nz+1,    nt+1]
        @test u[ny, ModeNumber(0,  nz, -nt)] ==      A[ny, 1, nz+1, Nt-nt+1]
        @test u[ny, ModeNumber(0, -nz,  nt)] == conj(A[ny, 1, nz+1, Nt-nt+1])
        @test u[ny, ModeNumber(0, -nz, -nt)] == conj(A[ny, 1, nz+1,    nt+1])
    end
    for ny in 1:Ny, nx in 1:(Nx >> 1), nz in -(Nz >> 1):(Nz >> 1), nt in (Nt >> 1):(Nt >> 1)
        _nz = nz >= 0 ? nz+1 : Nz+nz+1
        _nt = nt >= 0 ? nt+1 : Nt+nt+1
        @test u[ny, ModeNumber( nx, nz, nt)] == A[ny, nx+1,         _nz,      _nt]
        _nz = nz <= 0 ? -nz+1 : Nz-nz+1
        _nt = nt <= 0 ? -nt+1 : Nt-nt+1
        @test u[ny, ModeNumber(-nx, nz, nt)] == conj(A[ny, nx+1, _nz, _nt])
    end
end
