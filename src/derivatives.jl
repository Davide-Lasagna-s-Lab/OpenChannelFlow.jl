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
    @loop_modes S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nz, _nt] = 1im*nz*β*u[ny, _nz, _nt]
    end
    return out
end

function d2dx32!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    @loop_modes S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nz, _nt] = -(nz*β)^2*u[ny, _nz, _nt]
    end
    return out
end

function laplacian!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    d2dx22!(out, u)
    @loop_modes S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nz, _nt] += -(nz*β)^2*u[ny, _nz, _nt]
    end
    return out
end

function dds!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    @loop_modes S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nz, _nt] = 1im*nt*u[ny, _nz, _nt]
    end
    return out
end
function dds!(out::ProjectedField{G, M}, a::ProjectedField{G, M}) where {S, G<:ChannelGrid{S}, M}
    @loop_modes S[3] S[2] for m in 1:M
        @inbounds out[m, _nz, _nt] = 1im*nt*a[m, _nz, _nt]
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
