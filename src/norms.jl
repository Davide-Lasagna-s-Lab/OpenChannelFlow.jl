# Norm definitions and special scaling for channel flow fields.

# ----------------------- #
# standard inner products #
# ----------------------- #
# TODO: replace loop with macro that doesn't repeat iterations
function LinearAlgebra.dot(u::SCField{G}, v::SCField{G}) where {S, G<:ChannelGrid{S}}
    sum = 0.0
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in -(S[2] >> 1):(S[2] >> 1), ny in 1:S[1]
        @inbounds sum += grid(u).ws[ny]*real(dot(u[ny, ModeNumber(nz, nt)], v[ny, ModeNumber(nz, nt)]))
    end
    return sum/2
end

function LinearAlgebra.dot(u::VectorField{N, <:SCField{G}}, v::VectorField{N, <:SCField{G}}) where {N, S, G<:ChannelGrid{S}}
    sum = 0.0
    for n in 1:N
        @inbounds sum += dot(u[n], v[n])
    end
    return sum
end

function LinearAlgebra.dot(a::ProjectedField{G, M}, b::ProjectedField{G, M}) where {S, G<:ChannelGrid{S}, M}
    sum = 0.0
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in -(S[2] >> 1):(S[2] >> 1), m in 1:M
        @inbounds sum += real(dot(a[m, ModeNumber(nz, nt)], b[m, ModeNumber(nz, nt)]))
    end
    return sum/2
end

LinearAlgebra.norm(u::Union{SCField, VectorField, ProjectedField}) = sqrt(dot(u, u))


# ----------- #
# other norms #
# ----------- #
function normdiff(u::SCField{G}, v::SCField{G}, shifts=(0, 0), tmp::SCField{G}=zero(v)) where {S, G<:ChannelGrid{S}}
    sum = 0.0
    tmp .= v
    shift!(tmp, shifts)
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in -(S[2] >> 1):(S[2] >> 1), ny in 1:S[1]
        @inbounds sum += grid(u).ws[ny]*abs2(u[ny, ModeNumber(nz, nt)] - tmp[ny, ModeNumber(nz, nt)])
    end
    return sqrt(sum/2)
end

function normdiff(u::VectorField{N, <:SCField{G}}, v::VectorField{N, <:SCField{G}}, shifts=(0, 0), tmp::SCField{G}=zero(u[1])) where {N, S, G<:ChannelGrid{S}}
    sum = 0.0
    for n in 1:N
        sum += normdiff(u[n], v[n], shifts, tmp)^2
    end
    return sqrt(sum)
end

function normdiff(a::ProjectedField{G, M}, b::ProjectedField{G, M}, shifts=(0, 0), tmp::ProjectedField{G, M}=zero(b)) where {S, G<:ChannelGrid{S}, M}
    sum = 0.0
    tmp .= b
    shift!(tmp, shifts)
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in -(S[2] >> 1):(S[2] >> 1), m in 1:M
        @inbounds sum += abs2(a[m, ModeNumber(nz, nt)] - tmp[m, ModeNumber(nz, nt)])
    end
    return sqrt(sum/2)
end

function minnormdiff(u::Union{SCField{G}, VectorField{D, <:SCField{G}}, ProjectedField{G}},
                     v::Union{SCField{G}, VectorField{D, <:SCField{G}}, ProjectedField{G}},
                     N::NTuple{2, Int}=(32, 32),
                  tmp1::SCField{G}=zero(v),
                  tmp2::SCField{G}=zero(v)) where {D, G}
    # minimum values
    min_diff = Inf
    sz_min   = Inf
    st_min   = Inf

    # get shift steps
    _sz = (2π/grid(u).β)/N[1]
    _st = 1/N[1]

    # loop over available z and t shifts
    tmp1 .= v
    for ti in 0:N[2] - 1
        for zi in 0:N[1] - 1
            diff = normdiff(u, tmp1, tmp2)
            if diff < min_diff
                min_diff = diff
                sz_min = _sz*zi
                st_min = _st*ti
            end
            shift!(tmp1, (_sz, 0))
        end
        shift!(tmp1, (0, _st))
    end

    return min_diff, (sz_min, st_min)
end
