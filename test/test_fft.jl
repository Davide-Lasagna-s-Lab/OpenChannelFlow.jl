@testset "FFT transforms                        " begin
    # construct grid
    Ny = 16; Nz = 33; Nt = 11
    g = ChannelGrid(chebpts(Ny), Nz, Nt,
                    1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny))

    # create plans
    plans  = @test_nowarn OpenChannelFlow.FFTPlans(g; dealias=false, flags=FFTW.ESTIMATE)
    plansd = @test_nowarn OpenChannelFlow.FFTPlans(g; dealias=true,  flags=FFTW.ESTIMATE)

    # randon signal
    U  = SCField(g, rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    Ud = growto(U, OpenChannelFlow._padded_size(Nz, Nt, Val(3/2)))
    u  = PCField(g)
    ud = PCField(g, dealias=true)

    # test transforms
    @test  plans(similar(U),  plans(u,  U))  ≈ U
    @test plansd(similar(U), plansd(ud, U)) ≈ U

    # test allocations
    fun(plan, A, B) = @allocated plan(A, B)
    @test fun(plans,  u,          U)  == 0
    @test fun(plans,  similar(U), u)  == 0
    @test fun(plansd, ud,         U)  == 0
    @test fun(plansd, similar(U), ud) == 0

    # test allocating transforms
    @test FFT(plans(similar(u), U)) ≈ U
    @test IFFT(U) == plans(similar(u), U)
    @test FFT(plans(similar(u), U), OpenChannelFlow._padded_size(Nz, Nt, Val(3/2))) ≈ Ud
    @test IFFT(U, OpenChannelFlow._padded_size(Nz, Nt, Val(3/2))) == plansd(similar(ud), U)
end
