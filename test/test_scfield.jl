@testset "Spectral channel field                " begin
    # construct grid
    Ny = 16; Nz = 33; Nt = 33
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
    for ny in 1:Ny
        @test u[ny, ModeNumber(0, 0)] == A[ny, 1, 1]
    end
    for ny in 1:Ny, nt in 1:(Nz >> 1)
        @test u[ny, ModeNumber(0,  nt)] == A[ny, 1, nt+1]
        @test u[ny, ModeNumber(0, -nt)] == A[ny, 1, Nt-nt+1]
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1)
        @test u[ny, ModeNumber( nz,  0)] ==      A[ny, nz+1, 1]
        @test u[ny, ModeNumber(-nz,  0)] == conj(A[ny, nz+1, 1])
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1), nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber( nz,  nt)] ==      A[ny, nz+1,    nt+1]
        @test u[ny, ModeNumber( nz, -nt)] ==      A[ny, nz+1, Nt-nt+1]
        @test u[ny, ModeNumber(-nz,  nt)] == conj(A[ny, nz+1, Nt-nt+1])
        @test u[ny, ModeNumber(-nz, -nt)] == conj(A[ny, nz+1,    nt+1])
    end

    # test growto!
    u_new = growto!(u, (65, 49))
    @test size(u_new) == (Ny, (65 >> 1) + 1, 49)
    @test u_new[:, 1:(Nz >> 1) + 1, 1:(Nt >> 1) + 1] == u[:, 1:(Nz >> 1) + 1, 1:(Nt >> 1) + 1]
    @test u_new[:, 1:(Nz >> 1) + 1, end - (Nt >> 1) + 1:end] == u[:, 1:(Nz >> 1) + 1, (Nt >> 1) + 2:Nt]
    @test all(u_new[:, 1:(Nz >> 1) + 1, (Nt >> 1) + 2:end - (Nt >> 1) - 1] .== 0)
    @test all(u_new[:, (Nz >> 1) + 2:end, :] .== 0)

    # test setindex using mode number
    Random.seed!(1)
    numb = randn(ComplexF64)
    u[1, ModeNumber(0, 0)] = numb
    @test u[1, 1, 1] == real(numb)
    nt = rand(1:(Nt >> 1))
    u[1, ModeNumber(0, nt)] = numb
    @test u[1, 1, nt+1] == conj(u[1, 1, Nt-nt+1]) == numb
    u[1, ModeNumber(0, -nt)] = numb
    @test u[1, 1, nt+1] == conj(u[1, 1, Nt-nt+1]) == conj(numb)
    nz = rand(1:(Nz >> 1))
    nt = rand(0:(Nt >> 1))
    u[1, ModeNumber(nz, nt)] = numb
    @test u[1, nz+1, nt+1] == numb
    nt = rand(-(Nt >> 1):-1)
    u[1, ModeNumber(nz, nt)] = numb
    @test u[1, nz+1, Nt+nt+1] == numb
    nz = rand(-(Nz >> 1):-1)
    nt = rand(0:(Nt >> 1))
    u[1, ModeNumber(nz, nt)] = numb
    @test u[1, -nz+1, Nt-nt+1] == conj(numb)
    nt = rand(-(Nt >> 1):-1)
    u[1, ModeNumber(nz, nt)] = numb
    @test u[1, -nz+1, -nt+1] == conj(numb)
end
