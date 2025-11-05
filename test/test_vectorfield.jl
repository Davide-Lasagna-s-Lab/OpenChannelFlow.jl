@testset "Vector field                          " begin
    # construct grid
    Ny = 16; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # test vectorfield construction
    u = @test_nowarn VectorField(g)
    @test u isa VectorField{3, SCField{typeof(g), Float64}}
    u = @test_nowarn VectorField(g, Float32)
    @test u isa VectorField{3, SCField{typeof(g), Float32}}
    u = @test_nowarn VectorField(g, N=2)
    @test u isa VectorField{2, SCField{typeof(g), Float64}}
    u = @test_nowarn VectorField(g, type=PCField)
    @test u isa VectorField{3, PCField{typeof(g), Float64}}
    u = @test_nowarn VectorField(g, type=PCField, dealias=false)
    @test u isa VectorField{3, PCField{typeof(g), Float64}}
    u = @test_nowarn VectorField(g, type=PCField, dealias=true)
    @test u isa VectorField{3, PCField{typeof(g), Float64}}
    @test size(u[1]) == (Ny, OpenChannelFlow._padded_size(Nz, Nt, Val(3/2))...)
    u = @test_nowarn VectorField(g, ((y, z, t)->1.0, (y, z, t)->(1 - y^2)*cos(2π*z)), 1.0)
    @test u isa VectorField{2, PCField{typeof(g), Float64}}
    @test all(u[1] .== 1.0)
    pts = points(g, 1.0)
    @test all(u[2] .== (1 .- reshape(pts[1], :, 1, 1).^2).*cos.(2π.*reshape(pts[2], 1, :, 1)))
    u = @test_nowarn VectorField(g, ((y, z, t)->1.0, (y, z, t)->(1 - y^2)*cos(2π*z)), 1.0, dealias=true)
    @test u isa VectorField{2, PCField{typeof(g), Float64}}
    @test all(u[1] .== 1.0)
    pts = points(g, 1.0, OpenChannelFlow._padded_size(Nz, Nt, Val(3/2)))
    @test all(u[2] .== (1 .- reshape(pts[1], :, 1, 1).^2).*cos.(2π.*reshape(pts[2], 1, :, 1)))

    # test interface
    @test size(u) == (2,)
    @test length(u) == 2
    @test eltype(u) == typeof(PCField(g))
    @test OpenChannelFlow.datatype(u) == Float64
    @test similar(u) isa VectorField{2, PCField{typeof(g), Float64}}
    @test copy(u)[1] == u[1]
    @test copy(u)[2] == u[2]
    @test all(zero(u)[1] .== 0.0)
    @test all(zero(u)[2] .== 0.0)

    # test cross product
    u = VectorField(g, ((y, z, t)->1.0, (y, z, t)->(1 - y^2)*cos(2π*z), (y, z, t)->-2.0), 1.0)
    ku = zero(u)
    mag = randn()
    OpenChannelFlow.cross_k!(ku, u, mag)
    @test parent(ku[1]) == -mag.*parent(u[2])
    @test parent(ku[2]) ==  mag.*parent(u[1])

    # test growto
    u = VectorField(g)
    for n in 1:3
        u[n] .= randn(ComplexF64, Ny, (Nz >> 1) + 1, Nt)
    end
    v = growto(u, (65, 49))
    for n in 1:3
        @test v[n] == growto(u[n], (65, 49))
    end
end
