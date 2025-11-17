# Norm weight applied to the dot product of projected fields

struct FarazmandWeight{T<:Real}
    ω::T
    β::T
end
Base.getindex(A::FarazmandWeight, ::Int, nz::Int, nt::Int) = 1/(1 + (A.β*nz)^2 + (A.ω*nt)^2)

function LinearAlgebra.mul!(a::ProjectedField{G, M, T}, A::FarazmandWeight{T}) where {S, G<:ChannelGrid{S}, M, T}
    @loop_modes S[3] S[2] for m in 1:M
        @inbounds a[m, _nz, _nt] *= A[m, nz, nt]
    end
    return a
end

function LinearAlgebra.dot(a::ProjectedField{G, M, T}, A::FarazmandWeight{T}, b::ProjectedField{G, M, T}) where {S, G<:ChannelGrid{S}, M, T}
    sum = zero(T)
    @loop_nt S[3] for m in 1:M
        @inbounds sum += A[m, 0, nt]*real(dot(a[m, 1, _nt], b[m, 1, _nt]))
    end
    @loop_nt S[3] for _nz in 2:(S[2] >> 1) + 1, m in 1:M
        @inbounds sum += 2*A[m, _nz-1, nt]*real(dot(a[m, _nz, _nt], b[m, _nz, _nt]))
    end
    return sum/2
end
