# Utility object to allow dispatch for different indexing methods on SCField

struct ModeNumber
    nz::Int
    nt::Int
end

function _convert_modenumber(n::ModeNumber, Nt)
    do_conj = false
    if n.nz >= 0
        _nz = n.nz + 1
        _nt = n.nt >= 0 ? n.nt + 1 : Nt + n.nt + 1
        (n.nz == 0 && n.nt < 0) && (do_conj = true)
    else
        _nz = -n.nz + 1
        _nt = n.nt > 0 ? Nt - n.nt + 1 : -n.nt + 1
        do_conj = true
    end
    return _nz, _nt, do_conj
end

# TODO: add macros for simpler looping over all wavenumbers
