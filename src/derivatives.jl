# Derivative methods for flow fields.

# ------------------------ #
# scalar field derivatives #
# ------------------------ #
function ddx1!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    α = grid(u).α
    @loop_modes S[4] S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nx, _nz, _nt] = 1im*nx*α*u[ny, _nx, _nz, _nt]
    end
    return out
end

ddx2!(out::F, u::F) where {F<:Union{PCField, SCField}} = mul!(out, grid(u).Dy,  u)

function ddx3!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    β = grid(u).β
    @loop_modes S[4] S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nx, _nz, _nt] = 1im*nz*β*u[ny, _nx, _nz, _nt]
    end
    return out
end

function laplacian!(out::SCField{G}, u::SCField{G}) where {S, G<:ChannelGrid{S}}
    # get domain sizes
    α = grid(u).α
    β = grid(u).β

    # take second y-derivative
    mul!(out, grid(u).Dy2, u)

    # take second x- and z-derivatives
    @loop_modes S[4] S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nx, _nz, _nt] -= ((nz*β)^2 + (nx*α)^2)*u[ny, _nx, _nz, _nt]
    end

    return out
end

function dds!(out::ProjectedField{G, M}, a::ProjectedField{G, M}) where {S, G<:ChannelGrid{S}, M}
    @loop_modes S[4] S[3] S[2] for m in 1:M
        @inbounds out[m, _nx, _nz, _nt] = 1im*nt*a[m, _nx, _nz, _nt]
    end
    return out
end


# ------------------------ #
# vector field derivatives #
# ------------------------ #
for name in [:ddx1!, :ddx2!, :ddx3!, :laplacian!]
    @eval begin
        function $name(out::VectorField{N}, u::VectorField{N}) where {N}
            for n in 1:N
                $name(out[n], u[n])
            end
            return out
        end
    end
end
