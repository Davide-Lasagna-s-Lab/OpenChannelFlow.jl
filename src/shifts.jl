# Symmetry operators on channel fields.

# TODO: add discrete symmetries: σ₁[u,v,w](y,z)->[u,v,-w](y,-z), σ₂[u,v,w](y,z)->[-u,-v,w](-y,z)
# see: Visualizing the geometry of state space in plane Couette flow, Gibson, Halcrow, Cvitanovic, section 2.2

"""
    shift!(u::F, shifts::NTuple{2, Real}) -> F

Return the channel field `u` shifted in spanwise and temporal directions.

Apply a continuous shift to the field `u` in both the spanwise and temporal
directions, given by `shifts[1]` and `shifts[2]` respectively. This function
modifies the field it is provided, so be careful when using.
"""
shift!(u::Union{SCField, VectorField, ProjectedField}, shifts) = tshift!(zshift!(u, shifts[1]), shifts[2])


# -------------- #
# spanwise shift #
# -------------- #
zshift!(u::SCField{G}, sz) where {S, G<:ChannelGrid{S}} = _perform_zshift!(u, sz, S[1], S[2], S[3])
zshift!(a::ProjectedField{G, M}, sz) where {S, G<:ChannelGrid{S}, M} = _perform_zshift!(a, sz, M, S[2], S[3])

function zshift!(u::VectorField{N, <:SCField}, sz) where {N}
    for n in 1:N
        zshift!(u[n], sz)
    end
    return u
end

@inline function _perform_zshift!(field, sz, S1, S2, S3)
    sz == 0 && return field
    for nz in 0:(S2 >> 1)
        val = cis(nz*sz*grid(field).β)
        for nt in -(S3 >> 1):(S3 >> 1), m in 1:S1
            @inbounds field[m, ModeNumber(nz, nt)] *= val
        end
    end
    return field
end


# -------------- #
# temporal shift #
# -------------- #
tshift!(u::SCField{G}, st) where {S, G<:ChannelGrid{S}} = _perform_tshift!(u, st, S[1], S[2], S[3])
tshift!(a::ProjectedField{G, M}, st) where {S, G<:ChannelGrid{S}, M} = _perform_tshift!(a, st, M, S[2], S[3])

function tshift!(u::VectorField{N, <:SCField}, st) where {N}
    for n in 1:N
        tshift!(u[n], st)
    end
    return u
end

# ? for some reason inlining this function improves performance, but inlining the
# ? spanwise version does nothing?
@inline function _perform_tshift!(field, st, S1, S2, S3)
    st == 0 && return field
    for nt in -(S3 >> 1):(S3 >> 1)
        val = cispi(2*nt*st)
        for nz in 0:(S2 >> 1), m in 1:S1
            @inbounds field[m, ModeNumber(nz, nt)] *= val
        end
    end
    return field
end
