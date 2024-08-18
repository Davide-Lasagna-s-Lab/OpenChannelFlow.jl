@testset "FFT Transforms                        " begin
    # randon signal
    Ny = 16; Nz = 33; Nt = 11
    Â = RPCF.apply_symmetry!(rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt))
    Â[:, 1, 1] .= real.(Â[:, 1, 1])
    A = zeros(Float64, Ny, Nz, Nt)
    A_dealiased = zeros(Float64, Ny, RPCF.padded_size(Nz, Nt, 3/2)...)
    B = zeros(ComplexF64, Ny, (Nz >> 1) + 1, Nt)

    # create plans
    plans = @test_nowarn RPCF.FFTPlans(Ny, Nz, Nt; dealias=false, flags=FFTW.ESTIMATE)
    plans_dealiased = @test_nowarn RPCF.FFTPlans(Ny, Nz, Nt; dealias=true, flags=FFTW.ESTIMATE)

    plans(A, Â)
    plans(B, A)
    @test Â ≈ B

    plans_dealiased(A_dealiased, Â)
    plans_dealiased(B, A_dealiased)
    @test Â ≈ B
end
