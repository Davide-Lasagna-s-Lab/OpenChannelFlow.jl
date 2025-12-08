# Implementation of the spectral representation of scalar channel fields

# -------------- #
# spectral field #
# -------------- #
struct FTField{G<:ChannelGrid, T} <: AbstractScalarField{4, Complex{T}}
    grid::G
    data::Array{Complex{T}, 4}

    function FTField(g::G, data::Array{Complex{T}, 4}) where {G, T}
        apply_symmetry!(data)
        data[:, 1, 1, 1] .= real.(data[:, 1, 1, 1])
        new{G, T}(g, data)
    end
end
FTField(g::G, ::Type{T}=Float64) where {S, G<:ChannelGrid{S}, T} = FTField(g, zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3], S[4]))

Base.parent(u::FTField)                                 = u.data
Base.eltype(::FTField{G, T}) where {G, T}               = Complex{T}
Base.similar(u::FTField, ::Type{T}=eltype(u)) where {T} = FTField(grid(u), similar(parent(u), T))

NSEBase.hsize(::FTField{<:ChannelGrid{S}}) where {S} = ((S[2] >> 1) + 1, S[3], S[4])


# ---------------- #
# indexing methods #
# ---------------- #
# mode number idexing
Base.@propagate_inbounds function Base.getindex(u::FTField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(u[ny, _nx, _nz, _nt]) : u[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::FTField{G, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[4] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
                @inbounds u[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds u[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end

Base.@propagate_inbounds function Base.getindex(a::ProjectedField{<:FTField{G}}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(a[ny, _nx, _nz, _nt]) : a[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{<:FTField{G}, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid{S}, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[3] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
                @inbounds a[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds a[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# ------------------------------ #
# vector field grid constructors #
# ------------------------------ #
NSEBase.VectorField(g::ChannelGrid, ::Type{T}=Float64; N::Int=3, type::Type{S}=FTField, kwargs...) where {T, S} = VectorField([type(g, T; kwargs...) for _ in 1:N]...)
NSEBase.VectorField(g::ChannelGrid, funcs, period, ::Type{T}=Float64; dealias::Bool=false) where {T} = VectorField([Field(g, f, period, T, dealias=dealias) for f in funcs]...)
NSEBase.add_base!(u::VectorField{N, <:FTField}, base) where {N} = (u[1][:, 1, 1, 1] .+= base; return u)


#---------------- #
# projected stuff #
#---------------- #
NSEBase.ProjectedField(::G, modes, ::Type{T}=Float64) where {S, G<:ChannelGrid{S}, T} = ProjectedField(FTField{G, T}, zeros(Complex{T}, size(modes, 2), (S[2] >> 1) + 1, S[3], S[4]), modes)

channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)

function NSEBase.project!(a::ProjectedField{F, T}, u::VectorField{N, F}) where {S, F<:FTField{<:ChannelGrid{S}}, N, T}
    a .= zero(T)
    @loop_modes S[4] S[3] S[2] for m in axes(a, 1), n in 1:N
        @views @inbounds a[m, _nx, _nz, _nt] += channel_int(modes(a)[(S[1]*(n - 1) + 1):S[1]*n, m, _nx, _nz, _nt], grid(u).ws, u[n][:, _nx, _nz, _nt], S[1])
    end
    return a
end
NSEBase.project(u::VectorField{N, FTField{G, T}}, modes) where {N, G, T} = project!(ProjectedField(grid(u), modes, T), u)

function NSEBase.expand!(u::VectorField{N, F}, a::ProjectedField{F}) where {N, S, F<:FTField{<:ChannelGrid{S}}}
    @loop_modes S[4] S[3] S[2] for n in 1:N
        @views @inbounds mul!(u[n][:, _nx, _nz, _nt], modes(a)[(S[1]*(n - 1) + 1):S[1]*n, :, _nx, _nz, _nt], a[:, _nx, _nz, _nt])
    end
    return u
end

function dds!(out::ProjectedField{F}, a::ProjectedField{F}) where {S, F<:FTField{<:ChannelGrid{S}}}
    @loop_modes S[4] S[3] S[2] for m in axes(a, 1)
        @inbounds out[m, _nx, _nz, _nt] = 1im*nt*a[m, _nx, _nz, _nt]
    end
    return out
end

function NSEBase.ProjectedNSE(g::ChannelGrid{S}, Re::Real, ::Type{T}=Float64; Ro::Real=0, base::Vector{T}=g.y, flags=FFTW.EXHAUSTIVE, adjoint=true) where {S, T}
    # construct operators
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g, T)               for _ in 1:3]...) for _ in 1:6]
    pcache = [VectorField([  Field(g, T, dealias=true) for _ in 1:3]...) for _ in 1:8]
    nl = CartesianPrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
    ln = CartesianPrimitiveLNSE(T(Re), T(Ro), plans, scache, pcache, adjoint)

    return ProjectedNSE(scache[1][1], nl, ln, base, T)
end


# --------------- #
# utility methods #
# --------------- #
grid(u::FTField) = u.grid
grid(u::VectorField) = grid(u[1])

function growto(u::FTField{G, T}, N::NTuple{3, Int}) where {S, G<:ChannelGrid{S}, T}
    out = FTField(growto(grid(u), N), T)
    for ny in 1:S[1], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[4] >> 1):(S[4] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end

function growto(u::VectorField{L, <:FTField}, N::NTuple{3, Int}) where {L}
    v = VectorField(growto(u[1], N), N=L)
    for n in 1:L
        parent(v[n]) .= parent(growto(u[n], N))
    end
    return v
end

function apply_symmetry!(u::AbstractArray{T, 4}) where {T}
    Ny, _, Nz, Nt = size(u)
    for nt in 2:Nt, nz in 2:(Nz >> 1) + 1, ny in 1:Ny
        av = _average_complex(u[ny, 1, nz, nt], u[ny, 1, end-nz+2, end-nt+2])
        u[ny, 1,     nz,       nt]   =      av
        u[ny, 1, end-nz+2, end-nt+2] = conj(av)
    end
    for nz in 2:(Nz >> 1) + 1, ny in 1:Ny
        av = _average_complex(u[ny, 1, nz, 1], u[ny, 1, end-nz+2, 1])
        u[ny, 1,     nz,   1] =      av
        u[ny, 1, end-nz+2, 1] = conj(av)
    end
    for nt in 2:(Nt >> 1) + 1, ny in 1:Ny
        av = _average_complex(u[ny, 1, 1, nt], u[ny, 1, 1, end-nt+2])
        u[ny, 1, 1,     nt]   =      av
        u[ny, 1, 1, end-nt+2] = conj(av)
    end
    return u
end

function _average_complex(z1, z2)
    _re = 0.5*(real(z1) + real(z2))
    _im = 0.5*(imag(z1) - imag(z2))
    return _re + 1im*_im
end
