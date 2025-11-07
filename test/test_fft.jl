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
    @test  plans(similar(U),  plans(u,  U)) ≈ U
    @test plansd(similar(U), plansd(ud, U)) ≈ U

    # test allocations
    fun(plan, A, B) = @allocated plan(A, B)
    fun(plansd, ud, U) # required to avoid allocation when first called (for some unknown reason)
    @test fun(plans,  u,          U)  == 0
    @test fun(plans,  similar(U), u)  == 0
    @test fun(plansd, ud,         U)  == 0
    @test fun(plansd, similar(U), ud) == 0

    # test allocating transforms
    @test FFT(plans(similar(u), U)) ≈ U
    @test IFFT(U) == plans(similar(u), U)
    @test FFT(plans(similar(u), U), OpenChannelFlow._padded_size(Nz, Nt, Val(3/2))) ≈ Ud
    @test IFFT(U, OpenChannelFlow._padded_size(Nz, Nt, Val(3/2))) == plansd(similar(ud), U)

    # vector field transforms
    U = VectorField(g)
    for n in 1:3
        parent(U[n]) .= randn(ComplexF64, Ny, (Nz >> 1) + 1, Nt)
        parent(U[n])[:, 1, 1] .= real.(parent(U[n])[:, 1, 1])
        OpenChannelFlow.apply_symmetry!(parent(U[n]))
    end
    u = VectorField(g, type=PCField)
    U_new = plans(similar(U), plans(u, U))
    for n in 1:3
        @test U_new[n] ≈ U[n]
    end
    @test IFFT(U) == u
    u_new = IFFT(U, (65, 49))
    for n in 1:3
        @test u_new[n] ≈ IFFT(growto(U[n], (65, 49)))
    end
    U_new = FFT(u)
    for n in 1:3
        @test U_new[n] ≈ U[n]
    end
    U_new = FFT(u, (65, 49))
    for n in 1:3
        @test U_new[n] ≈ growto(U[n], (65, 49))
    end
end
