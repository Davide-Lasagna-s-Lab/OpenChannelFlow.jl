@testset "Field grid                            " begin
    # generate random inputs
    Ny    = rand(3:50)
    Nz    = rand(3:50)
    Nt    = rand(3:50)
    randD = rand([-2, -1, 1, 2])
    y     = rand(Float64, Ny)
    D1    = rand(Float32, (Ny, Ny))
    D2    = rand(Float16, (Ny + randD, Ny + randD))
    D_sec = rand(Float32, (Ny, Ny))
    w1    = rand(Int128, Ny)
    w2    = rand(Float16, Ny + randD)
    β     = π
    T     = 10*rand()

    # test point generation
    for dealias in [false, true]
        pts = points(ChannelGrid(y, Nz, Nt, β, D1, D_sec, w1, dealias=dealias, flags=FFTW.ESTIMATE), T)
        @test pts[1] == y
        @test pts[2] ≈  range(0, 2π*(1 - 1/Nz), length = Nz)/β # precision differences in operations
        @test pts[3] ≈  range(0,    (1 - 1/Nt), length = Nt)*T # mean they aren't exactly equal
    end
end
