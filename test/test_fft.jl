@testset "FFT transforms                        " begin
    # randon signal
    Ny = 16; Nz = 33; Nt = 11
    Â = OpenChannelFlow.apply_symmetry!(rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    Â[:, 1, 1] .= real.(Â[:, 1, 1])
    A = zeros(Float64, Ny, Nz, Nt)
    A_dealiased = zeros(Float64, Ny, OpenChannelFlow.padded_size(Nz, Nt, 3/2)...)
    B = zeros(ComplexF64, Ny, (Nz >> 1) + 1, Nt)

    # create plans
    plans = @test_nowarn OpenChannelFlow.FFTPlans(Ny, Nz, Nt; dealias=false, flags=FFTW.ESTIMATE)
    plans_dealiased = @test_nowarn OpenChannelFlow.FFTPlans(Ny, Nz, Nt; dealias=true, flags=FFTW.ESTIMATE)

    plans(A, Â)
    plans(B, A)
    @test Â ≈ B

    plans_dealiased(A_dealiased, Â)
    plans_dealiased(B, A_dealiased)
    @test Â ≈ B

    fun(plan, A, B) = @allocated plan(A, B)
    @test fun(plans, A, Â) == 0
    @test fun(plans, B, A) == 0
    @test fun(plans_dealiased, A_dealiased, Â) == 0
    @test fun(plans_dealiased, B, A_dealiased) == 0
end
