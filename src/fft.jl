# Fourier transforms for the scalar field

# ---------------- #
# transform struct #
# ---------------- #
struct FFTPlans{G, T, DEALIAS, PLAN, IPLAN}
    plan::PLAN
    iplan::IPLAN
    spectral_cache::Array{Complex{T}, 3}

    function FFTPlans(         ::G,
                               ::Type{T}=Float64;
                        dealias::Bool=true,
                          flags::UInt32=FFTW.EXHAUSTIVE,
                      timelimit::Real=FFTW.NO_TIMELIMIT) where {S, G<:ChannelGrid{S}, T}
        # grid size
        Ny = S[1]
        Nz, Nt = _padded_size(S[2], S[3], dealias ? Val(3/2) : Val(1))

        # construct arrays
        spectral_array = zeros(Complex{T}, Ny, (Nz >> 1) + 1, Nt)
        physical_array = zeros(T, Ny, Nz, Nt)

        # construct plans
        plan  = FFTW.plan_rfft(physical_array, [2, 3], flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, Nz, [2, 3], flags=flags, timelimit=timelimit)

        new{G, T, dealias, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array)
    end
end


# ------------------- #
# in-place transforms #
# ------------------- #
function (f::FFTPlans{G, T, true})(û::SCField{G, T}, u::PCField{G, T}) where {G, T}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    copy_from_padded!(parent(û), f.spectral_cache)
    parent(û) .*= 1/prod(size(u)[2:3])
    return û
end

# special method that adds the result to the output rather than overwriting
function (f::FFTPlans{G, T, true})(û::SCField{G, T}, u::PCField{G, T}, ::Val{false}) where {G, T}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    f.spectral_cache .*= 1/prod(size(u)[2:3])
    copy_add_from_padded!(parent(û), f.spectral_cache)
    return û
end

function (f::FFTPlans{G, T, false})(û::SCField{G, T}, u::PCField{G, T}) where {G, T}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    parent(û) .= f.spectral_cache./prod(size(u)[2:3])
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

function (f::FFTPlans{G, T, true})(u::PCField{G, T}, û::SCField{G, T}) where {G, T}
    copy_to_padded!(apply_mask!(f.spectral_cache), parent(û))
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
    parent(û) .= rfft(parent(u), [2, 3])./(S[2]*S[3])
    return û
end

function FFT(u::PCField{G, T}, N) where {S, G<:ChannelGrid{S}, T}
    û = growto(SCField(grid(u), rfft(parent(u), [2, 3])./(S[2]*S[3])), N)
    return û
end

FFT(u::VectorField{L, P})    where {L, P<:PCField} = VectorField([FFT(u[n])    for n in 1:L]...)
FFT(u::VectorField{L, P}, N) where {L, P<:PCField} = VectorField([FFT(u[n], N) for n in 1:L]...)

function IFFT(û::SCField{G, T}) where {S, G<:ChannelGrid{S}, T}
    u = PCField(grid(û), T)
    parent(u) .= brfft(parent(û), S[2], [2, 3])
    return u
end

function IFFT(û::SCField{G, T}, N) where {G, T}
    u = PCField(growto(grid(û), N), T)
    parent(u) .= brfft(parent(growto(û, N)), N[1], [2, 3])
    return u
end

IFFT(u::VectorField{L, S})    where {L, S<:SCField} = VectorField([IFFT(u[n])    for n in 1:L]...)
IFFT(u::VectorField{L, S}, N) where {L, S<:SCField} = VectorField([IFFT(u[n], N) for n in 1:L]...)


# --------------- #
# utility methods #
# --------------- #
function copy_to_padded!(upad::Array{Complex{T}}, u::Array{Complex{T}}) where {T}
    Nz, Nt = size(u)[2:3]
    Nt_pad = size(upad, 3)
    if Nt > 1
        if Nt % 2 == 0
            @views upad[:, 1:Nz, 1:(((Nt - 1) >> 1) + 1)]                    .= u[:, :, 1:(((Nt - 1) >> 1) + 1)]
            @views upad[:, 1:Nz, (((Nt - 1) >> 1) + 2 + Nt_pad - Nt):Nt_pad] .= u[:, :, (((Nt - 1) >> 1) + 2):Nt]
        else
            @views upad[:, 1:Nz, 1:(((Nt - 1) >> 1) + 1)]                    .= u[:, :, 1:(((Nt - 1) >> 1) + 1)]
            @views upad[:, 1:Nz, (((Nt - 1) >> 1) + 2 + Nt_pad - Nt):Nt_pad] .= u[:, :, (((Nt - 1) >> 1) + 2):Nt]
        end
    else
        @views upad[:, 1:Nz, 1] .= u[:, :, 1]
    end
    return upad
end

function copy_from_padded!(u::Array{Complex{T}}, upad::Array{Complex{T}}) where {T}
    Nz, Nt = size(u)[2:3]
    Nt_pad = size(upad, 3)
    if Nt > 1
        if Nt % 2 == 0
            # FIXME: doesn't work for even grid numbers, should figure out why at some point
            @views u[:, :, 1:((Nt >> 1) + 1)]  .= upad[:, 1:Nz, 1:((Nt >> 1) + 1)]
            @views u[:, :, ((Nt >> 1) + 2):Nt] .= upad[:, 1:Nz, ((Nt >> 1) + 2 + Nt_pad - Nt):Nt_pad]
        else
            @views u[:, :, 1:((Nt >> 1) + 1)]  .= upad[:, 1:Nz, 1:((Nt >> 1) + 1)]
            @views u[:, :, ((Nt >> 1) + 2):Nt] .= upad[:, 1:Nz, ((Nt >> 1) + 2 + Nt_pad - Nt):Nt_pad]
        end
    else
        @views u[:, :, 1] .= upad[:, 1:Nz, 1]
    end
    return u
end

function copy_add_from_padded!(u::Array{Complex{T}}, upad::Array{Complex{T}}) where {T}
    Nz, Nt = size(u)[2:3]
    Nt_pad = size(upad, 3)
    if Nt > 1
        if Nt % 2 == 0
            # FIXME: doesn't work for even grid numbers, should figure out why at some point
            @views u[:, :, 1:((Nt >> 1) + 1)]  .+= upad[:, 1:Nz, 1:((Nt >> 1) + 1)]
            @views u[:, :, ((Nt >> 1) + 2):Nt] .+= upad[:, 1:Nz, ((Nt >> 1) + 2 + Nt_pad - Nt):Nt_pad]
        else
            @views u[:, :, 1:((Nt >> 1) + 1)]  .+= upad[:, 1:Nz, 1:((Nt >> 1) + 1)]
            @views u[:, :, ((Nt >> 1) + 2):Nt] .+= upad[:, 1:Nz, ((Nt >> 1) + 2 + Nt_pad - Nt):Nt_pad]
        end
    else
        @views u[:, :, 1] .= upad[:, 1:Nz, 1]
    end
    return u
end

function apply_symmetry!(u::AbstractArray{Complex{T}, 3}) where {T<:AbstractFloat}
    Ny, _, Nt = size(u)
    for nt in 2:(((Nt - 1) >> 1) + 1), ny in 1:Ny
        pos = u[ny, 1, nt]
        neg = u[ny, 1, end - nt + 2]
        _re = 0.5*(real(pos) + real(neg))
        _im = 0.5*(imag(pos) - imag(neg))
        u[ny, 1, nt] = _re + 1im*_im
        u[ny, 1, end - nt + 2] = _re - 1im*_im
    end
    return u
end

apply_mask!(upad::Array{Complex{T}}) where {T} = (upad .= 0.0; return upad)
