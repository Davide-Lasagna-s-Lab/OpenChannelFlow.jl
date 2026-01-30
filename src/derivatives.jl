# Derivative methods for flow fields.

# ------------------------ #
# scalar field derivatives #
# ------------------------ #
function ddx1!(out::FTField{G}, u::FTField{G}) where {S, G<:Abstract1DChannelGrid{S}}
    @loop_modes S[4] S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nx, _nz, _nt] = 1im*nx*grid(u).α*u[ny, _nx, _nz, _nt]
    end
    return out
end

ddx2!(out::F, u::F) where {F<:Union{Field, FTField}} = mul!(out, grid(u).Dy,  u)

function ddx3!(out::FTField{G}, u::FTField{G}) where {S, G<:Abstract1DChannelGrid{S}}
    @loop_modes S[4] S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nx, _nz, _nt] = 1im*nz*grid(u).β*u[ny, _nx, _nz, _nt]
    end
    return out
end

function laplacian!(out::FTField{G}, u::FTField{G}) where {S, G<:ChannelGrid{S}}
    # take second y-derivative
    mul!(out, grid(u).Dy2, u)

    # take second x- and z-derivatives
    @loop_modes S[4] S[3] S[2] for ny in 1:S[1]
        @inbounds out[ny, _nx, _nz, _nt] -= ((nz*grid(u).β)^2 + (nx*grid(u).α)^2)*u[ny, _nx, _nz, _nt]
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
