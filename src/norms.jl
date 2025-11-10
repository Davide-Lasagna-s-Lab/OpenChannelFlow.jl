# Norm definitions and special scaling for channel flow fields.

# ----------------------- #
# standard inner products #
# ----------------------- #
function LinearAlgebra.dot(u::SCField{G}, v::SCField{G}) where {S, G<:ChannelGrid{S}}
    sum = 0.0
    @loop_nt S[3] for ny in 1:S[1]
        @inbounds sum += grid(u).ws[ny]*real(dot(u[ny, 1, _nt], v[ny, 1, _nt]))
    end
    @loop_nt S[3] for _nz in 2:(S[2] >> 1) + 1, ny in 1:S[1]
        @inbounds sum += 2*grid(u).ws[ny]*real(dot(u[ny, _nz, _nt], v[ny, _nz, _nt]))
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
    @loop_nt S[3] for m in 1:M
        @inbounds sum += real(dot(a[m, 1, _nt], b[m, 1, _nt]))
    end
    @loop_nt S[3] for _nz in 2:(S[2] >> 1) + 1, m in 1:M
        @inbounds sum += 2*real(dot(a[m, _nz, _nt], b[m, _nz, _nt]))
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
    @loop_nt S[3] for ny in 1:S[1]
        @inbounds sum += grid(u).ws[ny]*abs2(u[ny, 1, _nt] - tmp[ny, 1, _nt])
    end
    @loop_nt S[3] for _nz in 2:(S[2] >> 1) + 1, ny in 1:S[1]
        @inbounds sum += 2*grid(u).ws[ny]*abs2(u[ny, _nz, _nt] - tmp[ny, _nz, _nt])
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
    @loop_nt S[3] for m in 1:M
        @inbounds sum += abs2(a[m, 1, _nt] - tmp[m, 1, _nt])
    end
    @loop_nt S[3] for _nz in 2:(S[2] >> 1) + 1, m in 1:M
        @inbounds sum += 2*abs2(a[m, _nz, _nt] - tmp[m, _nz, _nt])
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
