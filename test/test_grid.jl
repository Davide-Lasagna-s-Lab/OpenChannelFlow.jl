@testset "Field Grid                            " begin
    # generate random inputs
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    randD = rand([-2, -1, 1, 2])
    y = rand(Float64, Ny)
    D1 = rand(Float32, (Ny, Ny))
    D2 = rand(Float16, (Ny + randD, Ny + randD))
    D_sec = rand(Float32, (Ny, Ny))
    w1 = rand(Int128, Ny)
    w2 = rand(Float16, Ny + randD)
    ω = abs(randn())
    β = π

    # test point generation
    g1 = RPCFGrid(y, Nz, Nt, β, ω, D1, D_sec, w1)
    gpoints = points(g1)
    @test gpoints[1] == y
    @test gpoints[2] ≈ range(0, 2π*(1 - 1/Nz), length = Nz)/β # precision differences in operations
    @test gpoints[3] ≈ range(0, 2π*(1 - 1/Nt), length = Nt)/ω # mean they aren't exactly equal

    # test size of grid
    @test size(g1) == (Ny, Nz, Nt)
    @test volume(g1) == 8π/ω

    # test comparison
    g2 = RPCFGrid(y, Nz, Nt + 1, β, ω, D1, D_sec, w1)
    g3 = RPCFGrid(rand(Float64, Ny), Nz, Nt, β, ω, D1, D_sec, w1)
    g4 = RPCFGrid(y, Nz, Nt, β, ω, rand(Float32, (Ny, Ny)), D_sec, w1)
    @test g1 != g2
    @test g1 != g3
    @test g1 == g4
end
