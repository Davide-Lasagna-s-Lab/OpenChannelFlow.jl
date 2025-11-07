# Derivative methods for flow fields.

# ------------------------ #
# scalar field derivatives #
# ------------------------ #
function ddx1!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    throw(error("not implemented"))
end

function d2dx12!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    throw(error("not implemented"))
end

ddx2!(out::F, u::F) where {F<:Union{PCField, SCField}} = mul!(out, grid(u).Dy,  u)
d2dx22!(out::F, u::F) where {F<:Union{PCField, SCField}} = mul!(out, grid(u).Dy2, u)

function ddx3!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] = 1im*nz*β*u[ny, ModeNumber(nz, nt)]
    end
    return out
end

function d2dx32!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] = -(nz*β)^2*u[ny, ModeNumber(nz, nt)]
    end
    return out
end

function laplacian!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    d2dx22!(out, u)
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] += -(nz*β)^2*u[ny, ModeNumber(nz, nt)]
    end
    return out
end

function dds!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), ny in 1:S[1]
        @inbounds out[ny, ModeNumber(nz, nt)] = 1im*nt*u[ny, ModeNumber(nz, nt)]
    end
    return out
end
function dds!(out::ProjectedField{G, M}, a::ProjectedField{G, M}) where {S, G<:ChannelGrid{S}, M}
    for nt in -(S[3] >> 1):(S[3] >> 1), nz in 0:(S[2] >> 1), m in 1:M
        @inbounds out[m, ModeNumber(nz, nt)] = 1im*nt*a[m, ModeNumber(nz, nt)]
    end
    return out
end


# ------------------------ #
# vector field derivatives #
# ------------------------ #
for name in [:ddx1!, :d2dx12!, :ddx2!, :d2dx22!, :ddx3!, :d2dx32!, :laplacian!, :dds!]
    @eval begin
        function $name(out::VectorField{N}, u::VectorField{N}) where {N}
            for n in 1:N
                $name(out[n], u[n])
            end
            return out
        end
    end
end
