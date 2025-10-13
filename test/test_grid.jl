@testset "Field grid                            " begin
    # generate random inputs
    Ny  = rand(3:51)
    Nz  = rand(3:2:51)
    Nt  = rand(3:2:51)
    y   = rand(Float64, Ny)
    Dy  = rand(Float64, (Ny, Ny))
    Dy2 = rand(Float64, (Ny, Ny))
    ws  = rand(Float64, Ny)
    β   = π
    T   = 10*rand()

    # test point generation
    g = ChannelGrid(y, Nz, Nt, β, Dy, Dy2, ws)
    pts = points(g, T)
    @test pts[1] == y
    @test pts[2]  ≈ range(0, 2π*(1 - 1/Nz), length=Nz)/β # precision differences in operations
    @test pts[3]  ≈ range(0,    (1 - 1/Nt), length=Nt)*T # mean they aren't exactly equal
    Nz_new = rand(Nt+2:2:81)
    Nt_new = rand(Nt+2:2:81)
    pts = points(g, T, (Nz_new, Nt_new))
    @test pts[2]  ≈ range(0, 2π*(1 - 1/Nz_new), length=Nz_new)/β # precision differences in operations
    @test pts[3]  ≈ range(0,    (1 - 1/Nt_new), length=Nt_new)*T # mean they aren't exactly equal

    # test growto
    g = ChannelGrid(y, Nz, Nt, β, Dy, Dy2, ws)
    Nz_new = rand(Nt+2:2:81)
    Nt_new = rand(Nt+2:2:81)
    g_new = growto(g, (Nz_new, Nt_new))
    pts = points(g, 1.0)
    pts_new = points(g_new, 1.0)
    @test pts_new[1] == pts[1]
    @test pts_new[2]  ≈ range(0, 2π*(1 - 1/Nz_new), length=Nz_new)/β # precision differences in operations
    @test pts_new[3]  ≈ range(0,    (1 - 1/Nt_new), length=Nt_new)   # mean they aren't exactly equal
end
