# Fourier transforms for the scalar field

# ---------------- #
# transform struct #
# ---------------- #
struct FFTPlans{DEALIAS, T, Nz, Nt, PLAN, IPLAN}
    plan::PLAN
    iplan::IPLAN
    spectral_cache::Array{Complex{T}, 3}

    function FFTPlans(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64; dealias::Bool=true, pad::Float64=3/2, flags::UInt32=FFTW.EXHAUSTIVE, timelimit::Real=FFTW.NO_TIMELIMIT) where {T}
        # construct arrays
        if dealias
            Nz_pad, Nt_pad = padded_size(Nz, Nt, pad)
            spectral_array = zeros(Complex{T}, Ny, (Nz_pad >> 1) + 1, Nt_pad)
            physical_array = zeros(T, Ny, Nz_pad, Nt_pad)
        else
            spectral_array = zeros(Complex{T}, Ny, (Nz >> 1) + 1, Nt)
            physical_array = zeros(T, Ny, Nz, Nt)
        end

        # construct plans
        plan = FFTW.plan_rfft(physical_array, [2, 3], flags=flags, timelimit=timelimit)
        iplan = dealias ? FFTW.plan_brfft(spectral_array, Nz_pad, [2, 3], flags=flags, timelimit=timelimit) : FFTW.plan_brfft(spectral_array, Nz, [2, 3], flags=flags, timelimit=timelimit)

        new{dealias, T, size(physical_array)[2:3]..., typeof(plan), typeof(iplan)}(plan, iplan, spectral_array)
    end
end
get_plan_types(::FFTPlans{DEALIAS, T, Nz, Nt, PLAN, IPLAN}) where {DEALIAS, T, Nz, Nt, PLAN, IPLAN} = (PLAN, IPLAN)
get_array_sizes(::FFTPlans{DEALIAS, T, Nz, Nt}) where {DEALIAS, T, Nz, Nt} = Nz, Nt


# ---------------------- #
# transformation methods #
# ---------------------- #
function (f::FFTPlans{true, T})(U::Array{Complex{T}}, u::Array{T}) where {T}
    FFTW.unsafe_execute!(f.plan, u, f.spectral_cache)
    copy_from_padded!(U, f.spectral_cache)
    U .*= 1/prod(size(u)[2:3])
    return U
end

function (f::FFTPlans{false, T})(U::Array{Complex{T}}, u::Array{T}) where {T}
    FFTW.unsafe_execute!(f.plan, u, f.spectral_cache)
    U .= f.spectral_cache./prod(size(u)[2:3])
    return U
end

function (f::FFTPlans{true, T})(u::Array{T}, U::Array{Complex{T}}) where {T}
    copy_to_padded!(apply_mask!(f.spectral_cache), U)
    FFTW.unsafe_execute!(f.iplan, f.spectral_cache, u)
    return u
end

function (f::FFTPlans{false, T})(u::Array{T}, U::Array{Complex{T}}) where {T}
    f.spectral_cache .= U
    FFTW.unsafe_execute!(f.iplan, f.spectral_cache, u)
    return u
end


# --------------- #
# utility methods #
# --------------- #
function padded_size(Nz, Nt, factor)
    Nz_pad = ceil(Int, Nz*factor)
    Nt_pad = ceil(Int, Nt*factor)
    Nz_pad = (Nz_pad - Nz) % 2 == 0 ? Nz_pad : Nz_pad + 1
    Nt_pad = (Nt_pad - Nt) % 2 == 0 ? Nt_pad : Nt_pad + 1
    return Nz_pad, Nt_pad
end

function copy_to_padded!(upad::Array{Complex{T}}, u::Array{Complex{T}}) where {T}
    Nz, Nt = size(u)[2:3]
    Nt_pad = size(upad, 3)
    if Nt > 1
        if Nt % 2 == 0
            @views copyto!(upad[:, 1:Nz, 1:(((Nt - 1) >> 1) + 1)], u[:, :, 1:(((Nt - 1) >> 1) + 1)])
            @views copyto!(upad[:, 1:Nz, (((Nt - 1) >> 1) + 2 + Nt_pad - Nt):Nt_pad], u[:, :, (((Nt - 1) >> 1) + 2):Nt])
        else
            @views copyto!(upad[:, 1:Nz, 1:(((Nt - 1) >> 1) + 1)], u[:, :, 1:(((Nt - 1) >> 1) + 1)])
            @views copyto!(upad[:, 1:Nz, (((Nt - 1) >> 1) + 2 + Nt_pad - Nt):Nt_pad], u[:, :, (((Nt - 1) >> 1) + 2):Nt])
        end
    else
        @views copyto!(upad[:, 1:Nz, 1], u[:, :, 1])
    end
    return upad
end

function copy_from_padded!(u::Array{Complex{T}}, upad::Array{Complex{T}}) where {T}
    Nz, Nt = size(u)[2:3]
    Nt_pad = size(upad, 3)
    if Nt > 1
        if Nt % 2 == 0
            # FIXME: doesn't work for even grid numbers, should figure out why at some point
            @views copyto!(u[:, :, 1:((Nt >> 1) + 1)], upad[:, 1:Nz, 1:((Nt >> 1) + 1)])
            @views copyto!(u[:, :, ((Nt >> 1) + 2):Nt], upad[:, 1:Nz, ((Nt >> 1) + 2 + Nt_pad - Nt):Nt_pad])
        else
            @views copyto!(u[:, :, 1:((Nt >> 1) + 1)], upad[:, 1:Nz, 1:((Nt >> 1) + 1)])
            @views copyto!(u[:, :, ((Nt >> 1) + 2):Nt], upad[:, 1:Nz, ((Nt >> 1) + 2 + Nt_pad - Nt):Nt_pad])
        end
    else
        @views copyto!(u[:, :, 1], upad[:, 1:Nz, 1])
    end
    return u
end

function apply_symmetry!(u::Array{Complex{T}, 3}) where {T<:AbstractFloat}
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
