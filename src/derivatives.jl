# Derivative methods for flow fields.

# ------------------- #
# spatial derivatives #
# ------------------- #
ddy!(out::F,   u::F) where {F<:Union{PCField, SCField}} = mul!(out, grid(u).Dy,  u)
function ddy!(out::VectorField{N, F}, u::VectorField{N, F}) where {N, F<:Union{PCField, SCField}}
    for n in 1:N
        ddy!(out[n], u[n])
    end
    return out
end

d2dy2!(out::F, u::F) where {F<:Union{PCField, SCField}} = mul!(out, grid(u).Dy2, u)
function d2dy2!(out::VectorField{N, F}, u::VectorField{N, F}) where {N, F<:Union{PCField, SCField}}
    for n in 1:N
        d2dy2!(out[n], u[n])
    end
    return out
end

function ddz!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] = 1im*nz*β*u[ny, ModeNumber(nz, nt)]
    end
    return out
end
function ddz!(out::VectorField{N, F}, u::VectorField{N, F}) where {N, F<:SCField}
    for n in 1:N
        ddz!(out[n], u[n])
    end
    return out
end

function d2dz2!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] = -(nz*β)^2*u[ny, ModeNumber(nz, nt)]
    end
    return out
end
function d2dz2!(out::VectorField{N, F}, u::VectorField{N, F}) where {N, F<:SCField}
    for n in 1:N
        d2dz2!(out[n], u[n])
    end
    return out
end

function laplacian!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    d2dy2!(out, u)
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] += -(nz*β)^2*u[ny, ModeNumber(nz, nt)]
    end
    return out
end
function laplacian!(out::VectorField{N, F}, u::VectorField{N, F}) where {N, F<:SCField}
    for n in 1:N
        laplacian!(out[n], u[n])
    end
    return out
end


# --------------- #
# time derivative #
# --------------- #
function dds!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] = 1im*nt*u[ny, ModeNumber(nz, nt)]
    end
    return out
end
function dds!(out::VectorField{N, F}, u::VectorField{N, F}) where {N, F<:SCField}
    for n in 1:N
        dds!(out[n], u[n])
    end
    return out
end
function dds!(out::ProjectedField{G, M}, a::ProjectedField{G, M}) where {S, G<:ChannelGrid{S}, M}
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), m in 1:M
        @inbounds out[m, ModeNumber(nz, nt)] = 1im*nt*a[m, ModeNumber(nz, nt)]
    end
    return out
end
