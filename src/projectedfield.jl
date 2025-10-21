# Coefficient array representing a projected channel field

# ----------------------- #
# projected channel field #
# ----------------------- #
struct ProjectedField{G, M, T, A} <: AbstractArray{Complex{T}, 3}
    data::Array{Complex{T}, 3}
    modes::A

    ProjectedField{G}(data::Array{Complex{T}}, modes::A) where {G, T, A} = new{G, size(data, 1), T, A}(data, modes)
end
ProjectedField(::Type{G}, modes, ::Type{T}=Float64) where {S, G<:ChannelGrid{S}, T} = ProjectedField{G}(zeros(Complex{T}, no_of_modes(modes), (S[2] >> 1) + 1, S[3]), modes)
ProjectedField(g::ChannelGrid, modes, ::Type{T}=Float64) where {T} = ProjectedField(typeof(g), modes, T)


# --------------- #
# utility methods #
# --------------- #
no_of_modes(modes::Array{Complex{T}, 4}) where {T} = size(modes, 2)
modes(a::ProjectedField) = a.modes


# ----------------- #
# interface methods #
# ----------------- #
Base.IndexStyle(::Type{<:ProjectedField})                            = Base.IndexLinear()
Base.parent(a::ProjectedField)                                       = a.data
Base.eltype(::ProjectedField{G, M, T}) where {G, M, T}               = Complex{T}
Base.size(a::ProjectedField)                                         = size(parent(a))
Base.similar(a::ProjectedField{G}, ::Type{T}=eltype(a)) where {G, T} = ProjectedField{G}(similar(parent(a), T), modes(a))
Base.copy(a::ProjectedField{G}) where {G}                            = ProjectedField{G}(copy(parent(a)), modes(a))
Base.zero(a::ProjectedField{G}) where {G}                            = ProjectedField{G}(zero(parent(a)), modes(a))
Base.abs(a::ProjectedField)                                          = (b = zero(a); b .= abs.(a); return b)

# ---------------- #
# indexing methods #
# ---------------- #
# linear indexing
Base.@propagate_inbounds function Base.getindex(u::ProjectedField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds val = parent(u)[i]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::ProjectedField, val, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = val
    return val
end

# mode number idexing
Base.@propagate_inbounds function Base.getindex(u::ProjectedField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nz, _nt, do_conj = _convert_modenumber(n, S[3])
    @boundscheck checkbounds(u, ny, _nz, _nt)
    @inbounds val = do_conj ? conj(u[ny, _nz, _nt]) : u[ny, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::ProjectedField{G, M, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}, M, T}
    _nz, _nt, do_conj = _convert_modenumber(n, S[3])
    _nt_sym = _nt != 1 ? S[3] - _nt + 2 : _nt
    val = (_nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(u, ny, _nz, _nt)
                @inbounds u[ny, _nz, _nt]     = do_conj ? conj(val) :      val
    _nz == 1 && @inbounds u[ny, _nz, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# ------------------ #
# conversion methods #
# ------------------ #
channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)

function project!(a::ProjectedField{G, M}, u::VectorField{N, <:SCField{G}}) where {S, G<:ChannelGrid{S}, M, N}
    a .= 0.0
    for nt in 1:S[3], nz in 1:(S[2] >> 1) + 1, m in 1:M, n in 1:N
        @views @inbounds a[m, nz, nt] += channel_int(modes(a)[(S[1]*(n - 1) + 1):S[1]*n, m, nz, nt], grid(u).ws, u[n][:, nz, nt], S[1])
    end
    return a
end
project(u::VectorField{N, <:SCField{G}}, modes) where {N, G} = project!(ProjectedField(G, modes), u)

function expand!(u::VectorField{N, <:SCField{G}}, a::ProjectedField{G, M}) where {N, S, G<:ChannelGrid{S}, M}
    for n in 1:N, nt in 1:S[3], nz in 1:(S[2] >> 1) + 1
        @views @inbounds mul!(u[n][:, nz, nt], modes(a)[(S[1]*(n - 1) + 1):S[1]*n, :, nz, nt], a[:, nz, nt])
    end
    return u
end
