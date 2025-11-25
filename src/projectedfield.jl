# Coefficient array representing a projected channel field.

# ----------------------- #
# projected channel field #
# ----------------------- #
struct ProjectedField{G, M, T, A} <: AbstractArray{Complex{T}, 4}
    grid::G
    data::Array{Complex{T}, 4}
    modes::A

    function ProjectedField(g::G, data::Array{Complex{T}, 4}, modes::A) where {G, T, A}
        _check_compatibility(g, data, modes)
        new{G, size(data, 1), T, A}(g, data, modes)
    end
end
ProjectedField(g::ChannelGrid{S}, modes, ::Type{T}=Float64) where {S, T} = ProjectedField(g, zeros(Complex{T}, no_of_modes(modes), (S[2] >> 1) + 1, S[3], S[4]), modes)


# --------------- #
# utility methods #
# --------------- #
no_of_modes(modes::Array{Complex{T}, 5}) where {T} = size(modes, 2)
modes(a::ProjectedField) = a.modes
grid(a::ProjectedField) = a.grid

function _check_compatibility(::ChannelGrid{S}, data, modes) where {S}
    (size(modes, 1) % S[1] == 0 &&
    (S[2] >> 1) + 1 == size(modes, 3) == size(data, 2)  &&
     S[3]           == size(modes, 4) == size(data, 3)  &&
     S[4]           == size(modes, 5) == size(data, 4)) || throw(ArgumentError("grid, data, and/or modes are not compatible sizes"))
    size(data, 1) == no_of_modes(modes) || throw(ArgumentError("number of modes available not compatible with data"))
    return nothing
end


# ----------------- #
# interface methods #
# ----------------- #
Base.IndexStyle(::Type{<:ProjectedField})                            = Base.IndexLinear()
Base.parent(a::ProjectedField)                                       = a.data
Base.eltype(::ProjectedField{G, M, T}) where {G, M, T}               = Complex{T}
Base.size(a::ProjectedField)                                         = size(parent(a))
Base.similar(a::ProjectedField{G}, ::Type{T}=eltype(a)) where {G, T} = ProjectedField(grid(a), similar(parent(a), T), modes(a))
Base.copy(a::ProjectedField{G}) where {G}                            = ProjectedField(grid(a), copy(parent(a)), modes(a))
Base.zero(a::ProjectedField{G}) where {G}                            = ProjectedField(grid(a), zero(parent(a)), modes(a))
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
Base.@propagate_inbounds function Base.getindex(a::ProjectedField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(a[ny, _nx, _nz, _nt]) : a[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{G, M, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}, M, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[3] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
                @inbounds a[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds a[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# ------------------ #
# conversion methods #
# ------------------ #
channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)

function project!(a::ProjectedField{G, M, T}, u::VectorField{N, <:SCField{G}}) where {S, G<:ChannelGrid{S}, M, N, T}
    a .= zero(T)
    @loop_modes S[4] S[3] S[2] for m in 1:M, n in 1:N
        @views @inbounds a[m, _nx, _nz, _nt] += channel_int(modes(a)[(S[1]*(n - 1) + 1):S[1]*n, m, _nx, _nz, _nt], grid(u).ws, u[n][:, _nx, _nz, _nt], S[1])
    end
    return a
end
project(u::VectorField{N, SCField{G, T}}, modes) where {N, G, T} = project!(ProjectedField(grid(u), modes, T), u)

function expand!(u::VectorField{N, <:SCField{G}}, a::ProjectedField{G, M}) where {N, S, G<:ChannelGrid{S}, M}
    @loop_modes S[4] S[3] S[2] for n in 1:N
        @views @inbounds mul!(u[n][:, _nx, _nz, _nt], modes(a)[(S[1]*(n - 1) + 1):S[1]*n, :, _nx, _nz, _nt], a[:, _nx, _nz, _nt])
    end
    return u
end
expand(a::ProjectedField{G, M, T}) where {S, G<:ChannelGrid{S}, M, T} = expand!(VectorField(grid(a), T, N=size(modes(a), 1)÷S[1]), a)
