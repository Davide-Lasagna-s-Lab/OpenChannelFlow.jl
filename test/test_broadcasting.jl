@testset "Field broadcasting                    " begin
    # construct grid
    Ny = 16; Nx = 15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    2π, 2π,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # test scalar broadcasting
    u = SCField(g, ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt))
    v = PCField(g, (y, x, z, t)->1.0, 1.0)
    w = VectorField(copy(u), copy(u))
    x = VectorField(copy(v), copy(v))
    a = ProjectedField(g, randn(ComplexF64, 3*Ny, 2, (Nx >> 1) + 1, Nz, Nt))
    parent(a) .= ones(2, (Nx >> 1) + 1, Nz, Nt)
    @test u .* 2 == SCField(g, 2*ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt))
    @test u .+ 2 == SCField(g,   ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt) .+ 2)
    @test v .* 2 == PCField(g, 2*ones(Float64,    Ny,  Nx,           Nz, Nt))
    @test v .+ 2 == PCField(g, 3*ones(Float64,    Ny,  Nx,           Nz, Nt))
    @test w .* 2 == VectorField(SCField(g, 2*ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)),      SCField(g, 2*ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)))
    @test w .+ 2 == VectorField(SCField(g,   ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)) .+ 2, SCField(g,   ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)) .+ 2)
    @test x .* 2 == VectorField(PCField(g, 2*ones(Float64,    Ny,  Nx,           Nz, Nt)),      PCField(g, 2*ones(Float64,    Ny,  Nx,           Nz, Nt)))
    @test x .+ 2 == VectorField(PCField(g,   ones(Float64,    Ny,  Nx,           Nz, Nt)) .+ 2, PCField(g,   ones(Float64,    Ny,  Nx,           Nz, Nt)) .+ 2)
    @test a .+ 2 == ProjectedField(g, 3*ones(ComplexF64, 2, (Nx >> 1) + 1, Nz, Nt), modes(a))
    @test a .* 2 == ProjectedField(g, 2*ones(ComplexF64, 2, (Nx >> 1) + 1, Nz, Nt), modes(a))

    # test vector addition
    vec = randn(Ny)
    outu = SCField(g, ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt))
    outv = PCField(g, ones(ComplexF64, Ny,  Nx          , Nz, Nt))
    for nx in 1:(Nx >> 1) + 1, nz in 1:Nz, nt in 1:Nt
        outu[:, nx, nz, nt] .+= vec
    end
    for nx in 1:Nx, nz in 1:Nz, nt in 1:Nt
        outv[:, nx, nz, nt] .+= vec
    end
    @test u .+ vec isa typeof(u)
    @test v .+ vec isa typeof(v)
    @test u .+ vec == outu
    @test v .+ vec == outv

    # test allocations
    @noinline fun(a, b, c) = @allocated a .= b .+ c.*2
    u = SCField(g,   ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt))
    v = SCField(g, 2*ones(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt))
    w = SCField(g)
    @test fun(w, u, v) == 0
    u = PCField(g, ones(Float64, Ny, Nx, Nz, Nt))
    v = PCField(g, ones(Float64, Ny, Nx, Nz, Nt))
    w = PCField(g)
    @test fun(w, u, v) == 0
    u = VectorField(g)
    v = VectorField(g)
    w = VectorField(g)
    @test fun(w, u, v) == 0
    u = VectorField(g, ((y, x, z, t)->1.0,), 1.0)
    v = VectorField(g, ((y, x, z, t)->(1 - y^2)*cos(2π*z)*atan(sin(2π*t)),), 1.0)
    w = VectorField(g, N=1, type=PCField)
    @test fun(w, u, v) == 0
    u = ProjectedField(g, randn(ComplexF64, 3*Ny, 2, (Nx >> 1) + 1, Nz, Nt))
    v = ProjectedField(g, randn(ComplexF64, 3*Ny, 2, (Nx >> 1) + 1, Nz, Nt))
    w = ProjectedField(g, randn(ComplexF64, 3*Ny, 2, (Nx >> 1) + 1, Nz, Nt))
    @test fun(w, u, v) == 0
end
