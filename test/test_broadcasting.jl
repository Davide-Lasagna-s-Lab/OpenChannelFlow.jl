@testset "Field broadcasting                    " begin
    # construct grid
    Ny = 16; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    2π,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # test scalar broadcasting
    u = SCField(g, ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    v = PCField(g, (y, z, t)->1.0, 1.0)
    w = VectorField(copy(u), copy(u))
    x = VectorField(copy(v), copy(v))
    a = ProjectedField(g, randn(ComplexF64, Ny, 2, (Nz >> 1) + 1, Nt))
    parent(a) .= ones(2, (Nz >> 1) + 1, Nt)
    @test u .* 2 == SCField(g, 2*ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    @test u .+ 2 == SCField(g,   ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt) .+ 2)
    @test v .* 2 == PCField(g, 2*ones(Float64,    Ny,  Nz,           Nt))
    @test v .+ 2 == PCField(g, 3*ones(Float64,    Ny,  Nz,           Nt))
    @test w .* 2 == VectorField(SCField(g, 2*ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt)),      SCField(g, 2*ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt)))
    @test w .+ 2 == VectorField(SCField(g,   ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt)) .+ 2, SCField(g,   ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt)) .+ 2)
    @test x .* 2 == VectorField(PCField(g, 2*ones(Float64,    Ny,  Nz,           Nt)),      PCField(g, 2*ones(Float64,    Ny,  Nz,           Nt)))
    @test x .+ 2 == VectorField(PCField(g,   ones(Float64,    Ny,  Nz,           Nt)) .+ 2, PCField(g,   ones(Float64,    Ny,  Nz,           Nt)) .+ 2)
    @test a .+ 2 == ProjectedField{typeof(g)}(3*ones(ComplexF64, 2, (Nz >> 1) + 1, Nt), modes(a))
    @test a .* 2 == ProjectedField{typeof(g)}(2*ones(ComplexF64, 2, (Nz >> 1) + 1, Nt), modes(a))

    # test vector addition
    vec = randn(Ny)
    outu = SCField(g, ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    outv = PCField(g, ones(ComplexF64, Ny,  Nz          , Nt))
    for nz in 1:(Nz >> 1) + 1, nt in 1:Nt
        outu[:, nz, nt] .+= vec
    end
    for nz in 1:Nz, nt in 1:Nt
        outv[:, nz, nt] .+= vec
    end
    @test u .+ vec isa typeof(u)
    @test v .+ vec isa typeof(v)
    @test u .+ vec == outu
    @test v .+ vec == outv

    # test allocations
    @noinline fun(a, b, c) = @allocated a .= b .+ c.*2
    u = SCField(g,   ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    v = SCField(g, 2*ones(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    w = SCField(g)
    @test fun(w, u, v) == 0
    u = PCField(g, ones(Float64, Ny, Nz, Nt))
    v = PCField(g, ones(Float64, Ny, Nz, Nt))
    w = PCField(g)
    @test fun(w, u, v) == 0
    u = VectorField(g)
    v = VectorField(g)
    w = VectorField(g)
    @test fun(w, u, v) == 0
    u = VectorField(g, ((y, z, t)->1.0,), 1.0)
    v = VectorField(g, ((y, z, t)->(1 - y^2)*cos(2π*z)*atan(sin(2π*t)),), 1.0)
    w = VectorField(g, N=1, type=PCField)
    @test fun(w, u, v) == 0
    u = ProjectedField(g, randn(ComplexF64, Ny, 2, (Nz >> 1) + 1, Nt))
    v = ProjectedField(g, randn(ComplexF64, Ny, 2, (Nz >> 1) + 1, Nt))
    w = ProjectedField(g, randn(ComplexF64, Ny, 2, (Nz >> 1) + 1, Nt))
    @test fun(w, u, v) == 0
end
