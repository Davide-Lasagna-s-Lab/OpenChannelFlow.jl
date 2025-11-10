# Utility object to allow dispatch for different indexing methods on SCField

# ?: I wonder if I could define a @modenumber macro that converts the indexes at parse time and keeps the nice loops?

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

macro loop_modes(Nt, Nz, expr)
    quote
        for $(esc(:_nt)) in 1:($(esc(Nt)) >> 1) + 1, $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1
            $(esc(:nz)) = $(esc(:_nz)) - 1
            $(esc(:nt)) = $(esc(:_nt)) - 1
            $(esc(expr))
        end
        for $(esc(:_nt)) in ($(esc(Nt)) >> 1) + 2:$(esc(Nt)), $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1
            $(esc(:nz)) = $(esc(:_nz)) - 1
            $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
            $(esc(expr))
        end
    end
end

macro loop_nt(Nt, expr)
    quote
        for $(esc(:_nt)) in 1:($(esc(Nt)) >> 1) + 1
            $(esc(:nt)) = $(esc(:_nt)) - 1
            $(esc(expr))
        end
        for $(esc(:_nt)) in ($(esc(Nt)) >> 1) + 2:$(esc(Nt))
            $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
            $(esc(expr))
        end
    end
end
