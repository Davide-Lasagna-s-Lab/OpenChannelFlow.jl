# Utility object to allow dispatch for different indexing methods on SCField

struct ModeNumber
    nz::Int
    nt::Int
end

function _convert_modenumber(n::ModeNumber, Nt)
    if n.nz >= 0
        _nz = n.nz + 1
        _nt = n.nt >= 0 ? n.nt + 1 : Nt + n.nt + 1
        do_conj = false
    else
        _nz = -n.nz + 1
        _nt = n.nt > 0 ? Nt - n.nt + 1 : -n.nt + 1
        do_conj = true
    end
    return _nz, _nt, do_conj
end

# TODO: loop macro with this signature
# @loop_modes begin
#     for ny in 1:Ny

#     end
# end
