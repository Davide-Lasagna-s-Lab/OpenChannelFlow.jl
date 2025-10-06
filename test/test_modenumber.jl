@testset "Mode number index conversion          " begin
    Nt = 23 # Nt has to be odd
    Nz = 23 # Nz can be even or odd
    nzs = collect(0:(Nz >> 1) + 1)
    nts = [collect(0:(Nt >> 1)); collect(-(Nt >> 1):-1)]
    for _nz in 1:(Nz >> 1) + 1, _nt in 1:Nt
        n = OpenChannelFlow.ModeNumber(nzs[_nz], nts[_nt])
        do_conj = (_nz == 1 && _nt > (Nt >> 1) + 1) ? true : false
        @test OpenChannelFlow._convert_modenumber(n, Nt) == (_nz, _nt, do_conj)
    end
    nzs = collect(0:-1:-(Nz >> 1))
    nts = [[0]; collect(-1:-1:-(Nt >> 1)); collect((Nt >> 1:-1:1))]
    for _nz in 2:(Nz >> 1) + 1, _nt in 1:Nt
        n = OpenChannelFlow.ModeNumber(nzs[_nz], nts[_nt])
        @test OpenChannelFlow._convert_modenumber(n, Nt) == (_nz, _nt, true)
    end
end
