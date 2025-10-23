# Symmetry operators on channel fields.

# TODO: is there also a discrete symmetry shift I can come up with?
# TODO: add docs to these functions

shift!(u::Union{SCField, VectorField}, shifts) = tshift!(zshift!(u, shifts[1]), shifts[2])
shift!(u::ProjectedField, shifts, β) = tshift!(zshift!(u, shifts[1], β), shifts[2])

function zshift!(u::SCField{G}, sz) where {S, G<:ChannelGrid{S}}
    sz == 0 && return u
    for nz in 0:(S[2] >> 1)
        val = cis(nz*sz*grid(u).β)
        for nt in -(S[3] >> 1):(S[3] >> 1), ny in 1:S[1]
            @inbounds u[ny, ModeNumber(nz, nt)] *= val
        end
    end
    return u
end

function zshift!(u::VectorField{N, <:SCField}, sz) where {N}
    for n in 1:N
        zshift!(u[n], sz)
    end
    return u
end

function zshift!(a::ProjectedField{G, M}, sz, β) where {S, G<:ChannelGrid{S}, M}
    sz == 0 && return a
    for nz in 0:(S[2] >> 1)
        val = cis(nz*sz*β)
        for nt in -(S[3] >> 1):(S[3] >> 1), m in 1:M
            @inbounds a[m, ModeNumber(nz, nt)] *= val
        end
    end
    return a
end

function tshift!(u::SCField{G}, st) where {S, G<:ChannelGrid{S}}
    st == 0 && return u
    for nt in -(S[3] >> 1):(S[3] >> 1)
        val = cispi(2*nt*st)
        for nz in 0:(S[2] >> 1), ny in 1:S[1]
            @inbounds u[ny, ModeNumber(nz, nt)] *= val
        end
    end
    return u
end

function tshift!(u::VectorField{N, <:SCField}, st) where {N}
    for n in 1:N
        tshift!(u[n], st)
    end
    return u
end

function tshift!(a::ProjectedField{G, M}, st) where {S, G<:ChannelGrid{S}, M}
    st == 0 && return a
    for nt in -(S[3] >> 1):(S[3] >> 1)
        val = cispi(2*nt*st)
        for nz in 0:(S[2] >> 1), m in 1:M
            @inbounds a[m, ModeNumber(nz, nt)] *= val
        end
    end
    return a
end
