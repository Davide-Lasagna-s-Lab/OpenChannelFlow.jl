# method definitions for vector calculus and derivative operations

# TODO: optimise these

function ReSolverInterface.grad!(∇u::VectorField{3, <:RPCFField{S}}, u::RPCFField{S}) where {S}
    ∇u[1] .= 0.0
    ddy!(∇u[2], u)
    ddz!(∇u[3], u)
    return ∇u
end

function ReSolverInterface.divergence!(div_u::RPCFField{S}, u::VectorField{3, <:RPCFField{S}}) where {S}
    ddy!(div_u, u[2])
    ddz_add!(div_u, u[3])
    return div_u
end

function ReSolverInterface.laplacian!(Δu::RPCFField{S}, u::RPCFField{S}) where {S}
    d2dy2!(Δu, u)
    d2dz2_add!(Δu, u)
    return Δu
end

# FIXME: is there a way to not allocate an array here?
function ReSolverInterface.convection!(u∇v::VectorField{3, <:RPCFField{S}}, u::VectorField{3, <:RPCFField{S}}, v::VectorField{3, <:RPCFField{S}}) where {S}
    cache = RPCFField(grid(u))
    mult!(u∇v[1], u[2], ddy!(cache, v[1]))
    mult_add!(u∇v[1], u[3], ddz!(cache, v[1]))
    mult!(u∇v[2], u[2], ddy!(cache, v[2]))
    mult_add!(u∇v[2], u[3], ddz!(cache, v[2]))
    mult!(u∇v[3], u[2], ddy!(cache, v[3]))
    mult_add!(u∇v[3], u[3], ddz!(cache, v[3]))
    return u∇v
end

function ReSolverInterface.convection2!(∇uv::VectorField{3, <:RPCFField{S}}, u::VectorField{3, <:RPCFField{S}}, v::VectorField{3, <:RPCFField{S}}) where {S}
    cache = RPCFField(grid(u))
    ∇uv[1] .= 0.0
    mult!(∇uv[2], v[1], ddy!(cache, u[1]))
    mult_add!(∇uv[2], v[2], ddy!(cache, u[2]))
    mult_add!(∇uv[2], v[3], ddy!(cache, u[3]))
    mult!(∇uv[3], v[1], ddz!(cache, u[1]))
    mult_add!(∇uv[3], v[2], ddz!(cache, u[2]))
    mult_add!(∇uv[3], v[3], ddz!(cache, u[3]))
    return ∇uv
end
