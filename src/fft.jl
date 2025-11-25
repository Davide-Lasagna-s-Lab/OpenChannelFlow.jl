# Fourier transforms for the scalar field

# ---------------- #
# transform struct #
# ---------------- #
struct FFTPlans{G, T, DEALIAS, PLAN, IPLAN}
    plan::PLAN
    iplan::IPLAN
    spectral_cache::Array{Complex{T}, 4}

    function FFTPlans(::G,
                      ::Type{T}=Float64;
               dealias::Bool=true,
                 flags::UInt32=FFTW.EXHAUSTIVE,
             timelimit::Real=FFTW.NO_TIMELIMIT) where {S, G<:ChannelGrid{S}, T}
        # grid size
        Ny = S[1]
        Nx, Nz, Nt = _padded_size(S[2:4], dealias ? Val(3/2) : Val(1))

        # construct arrays
        spectral_array = zeros(Complex{T}, Ny, (Nx >> 1) + 1, Nz, Nt)
        physical_array = zeros(T, Ny, Nx, Nz, Nt)

        # construct plans
        plan  = FFTW.plan_rfft(physical_array,      [2, 3, 4], flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, Nx, [2, 3, 4], flags=flags, timelimit=timelimit)

        new{G, T, dealias, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array)
    end
end


# ------------------- #
# in-place transforms #
# ------------------- #
function (f::FFTPlans{G, T, true})(û::SCField{G, T}, u::PCField{G, T}) where {S, G<:ChannelGrid{S}, T}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    copy_from_padded!(parent(û), f.spectral_cache)
    parent(û) .*= 1/prod(size(u)[2:4])
    return û
end

# special method that adds the result to the output rather than overwriting
function (f::FFTPlans{G, T, true})(û::SCField{G, T}, u::PCField{G, T}, ::Val{false}) where {S, G<:ChannelGrid{S}, T}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    f.spectral_cache .*= 1/prod(size(u)[2:4])
    add_from_padded!(parent(û), f.spectral_cache)
    return û
end

function (f::FFTPlans{G, T, false})(û::SCField{G, T}, u::PCField{G, T}) where {S, G<:ChannelGrid{S}, T}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    parent(û) .= f.spectral_cache./prod(S[2:4])
    return û
end

function (f::FFTPlans)(û::VectorField{N, S}, u::VectorField{N, P}) where {N, S<:SCField, P<:PCField}
    for n in 1:N
        f(û[n], u[n])
    end
    return û
end

function (f::FFTPlans)(û::VectorField{N, S}, u::VectorField{N, P}, ::Val{false}) where {N, S<:SCField, P<:PCField}
    for n in 1:N
        f(û[n], u[n], Val(false))
    end
    return û
end

function (f::FFTPlans{G, T, true})(u::PCField{G, T}, û::SCField{G, T}) where {S, G<:ChannelGrid{S}, T}
    copy_to_padded!(apply_mask!(f.spectral_cache, S[2], S[3], S[4]), parent(û))
    FFTW.unsafe_execute!(f.iplan, f.spectral_cache, parent(u))
    return u
end

function (f::FFTPlans{G, T, false})(u::PCField{G, T}, û::SCField{G, T}) where {G, T}
    f.spectral_cache .= parent(û)
    FFTW.unsafe_execute!(f.iplan, f.spectral_cache, parent(u))
    return u
end

function (f::FFTPlans{G, T, false})(u::PCField{G, T}, û::SCField{G, T}, ::Val{false}) where {G, T}
    FFTW.unsafe_execute!(f.iplan, parent(û), parent(u))
    return u
end

function (f::FFTPlans)(u::VectorField{N, P}, û::VectorField{N, S}) where {N, P<:PCField, S<:SCField}
    for n in 1:N
        f(u[n], û[n])
    end
    return u
end


# --------------------- #
# allocating transforms #
# --------------------- #
function FFT(u::PCField{G, T}) where {S, G<:ChannelGrid{S}, T}
    û = SCField(grid(u), T)
    parent(û) .= rfft(parent(u), [2, 3, 4])
    û .*= 1/prod(S[2:4])
    return û
end

function FFT(u::PCField{G, T}, N) where {S, G<:ChannelGrid{S}, T}
    û = growto(SCField(grid(u), rfft(parent(u), [2, 3, 4])./prod(S[2:4])), N)
    return û
end

FFT(u::VectorField{L, P})    where {L, P<:PCField} = VectorField([FFT(u[n])    for n in 1:L]...)
FFT(u::VectorField{L, P}, N) where {L, P<:PCField} = VectorField([FFT(u[n], N) for n in 1:L]...)

function IFFT(û::SCField{G, T}) where {S, G<:ChannelGrid{S}, T}
    u = PCField(grid(û), T)
    parent(u) .= brfft(parent(û), S[2], [2, 3, 4])
    return u
end

function IFFT(û::SCField{G, T}, N) where {G, T}
    u = PCField(growto(grid(û), N), T)
    parent(u) .= brfft(parent(growto(û, N)), N[1], [2, 3, 4])
    return u
end

IFFT(u::VectorField{L, S})    where {L, S<:SCField} = VectorField([IFFT(u[n])    for n in 1:L]...)
IFFT(u::VectorField{L, S}, N) where {L, S<:SCField} = VectorField([IFFT(u[n], N) for n in 1:L]...)


# --------------- #
# utility methods #
# --------------- #
function copy_to_padded!(upad::Array{T, 4}, u::Array{T, 4}) where {T}
    Nx, Nz, Nt = size(u)[2:4]
    Nzp, Ntp = size(upad)[3:4]
    zsize = (Nz >> 1) + 1
    tsize = (Nt >> 1) + 1
    @views upad[:, 1:Nx,           1:zsize,           1:tsize] .= u[:, :,         1:zsize,         1:tsize]
    @views upad[:, 1:Nx, (Nzp-zsize+2):Nzp,           1:tsize] .= u[:, :, (Nz-zsize+2):Nz,         1:tsize]
    if Nt > 1
        @views upad[:, 1:Nx,           1:zsize, (Ntp-tsize+2):Ntp] .= u[:, :,         1:zsize, (Nt-tsize+2):Nt]
        @views upad[:, 1:Nx, (Nzp-zsize+2):Nzp, (Ntp-tsize+2):Ntp] .= u[:, :, (Nz-zsize+2):Nz, (Nt-tsize+2):Nt]
    end
    return upad
end

function copy_from_padded!(u::Array{T, 4}, upad::Array{T, 4}) where {T}
    Nx, Nz, Nt = size(u)[2:4]
    Nzp, Ntp = size(upad)[3:4]
    zsize = (Nz >> 1) + 1
    tsize = (Nt >> 1) + 1
    @views u[:, :,         1:zsize, 1:tsize] .= upad[:, 1:Nx,           1:zsize, 1:tsize]
    @views u[:, :, (Nz-zsize+2):Nz, 1:tsize] .= upad[:, 1:Nx, (Nzp-zsize+2):Nzp, 1:tsize]
    if Nt > 1
        @views u[:, :,         1:zsize, (Nt-tsize+2):Nt] .= upad[:, 1:Nx,           1:zsize, (Ntp-tsize+2):Ntp]
        @views u[:, :, (Nz-zsize+2):Nz, (Nt-tsize+2):Nt] .= upad[:, 1:Nx, (Nzp-zsize+2):Nzp, (Ntp-tsize+2):Ntp]
    end
    return u
end

function add_from_padded!(u::Array{T, 4}, upad::Array{T, 4}) where {T}
    Nx, Nz, Nt = size(u)[2:4]
    Nzp, Ntp = size(upad)[3:4]
    zsize = (Nz >> 1) + 1
    tsize = (Nt >> 1) + 1
    @views u[:, :,         1:zsize, 1:tsize] .+= upad[:, 1:Nx,           1:zsize, 1:tsize]
    @views u[:, :, (Nz-zsize+2):Nz, 1:tsize] .+= upad[:, 1:Nx, (Nzp-zsize+2):Nzp, 1:tsize]
    if Nt > 1
        @views u[:, :,         1:zsize, (Nt-tsize+2):Nt] .+= upad[:, 1:Nx,           1:zsize, (Ntp-tsize+2):Ntp]
        @views u[:, :, (Nz-zsize+2):Nz, (Nt-tsize+2):Nt] .+= upad[:, 1:Nx, (Nzp-zsize+2):Nzp, (Ntp-tsize+2):Ntp]
    end
    return u
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

function apply_mask!(upad::Array{T}, Nx, Nz, Nt) where {T}
    upad[:, (Nx >> 1)+2:end,            :,                           :]              .= zero(T)
    upad[:,            :,    (Nz >> 1)+2:end-(Nz >> 1),             1:(Nt >> 1)+1]   .= zero(T)
    upad[:,            :,    (Nz >> 1)+2:end-(Nz >> 1), end-(Nt >> 1):end]           .= zero(T)
    upad[:,            :,              1:end,             (Nt >> 1)+2:end-(Nt >> 1)] .= zero(T)
    return upad
end
