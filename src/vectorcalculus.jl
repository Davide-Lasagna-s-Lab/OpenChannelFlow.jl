# method definitions for vector calculus and derivative operations

function ReSolverInterface.grad!(∇u::VectorField{3, RPCFField{S}}, u::RPCFField{S}) where {S}
    ∇u[1] .= 0.0
    ddy!(∇u[2], u)
    ddz!(∇u[3], u)
    return ∇u
end

function ReSolverInterface.divergence!(div_u::RPCFField{S}, u::VectorField{3, RPCFField{S}}) where {S}
    ddy!(div_u, u[2])
    ddz_add!(div_u, u[3])
    return div_u
end

function ReSolverInterface.laplacian!(Δu::RPCFField{S}, u::RPCFField{S}) where {S}
    d2dy2!(Δu, u)
    d2dz2!(Δu, u)
    return Δu
end

struct Convection!{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    cache::RPCFField{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    Convection!(g::RPCFGrid{S, DM, DEALIAS, PAD, PLAN, IPLAN}) where {S, DM, DEALIAS, PAD, PLAN, IPLAN} = new{S, DM, DEALIAS, PAD, PLAN, IPLAN}(RPCFField(g))
end
function (f::Convection!{S})(u∇v::VectorField{3, RPCFField{S}}, u::VectorField{3, RPCFField{S}}, v::VectorField{3, RPCFField{S}}) where {S}
    mult_add!(u∇v[1], u[2], ddy!(f.cache[1], v[1]))
    mult_add!(u∇u[1], u[3], ddz!(f.cache[1], v[1]))
    mult_add!(u∇v[2], u[2], ddy!(f.cache[1], v[2]))
    mult_add!(u∇v[2], u[3], ddz!(f.cache[1], v[2]))
    mult_add!(u∇v[3], u[2], ddy!(f.cache[1], v[3]))
    mult_add!(u∇v[3], u[3], ddz!(f.cache[1], v[3]))
    return u∇v
end

struct Convection2!{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    cache::RPCFField{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    Convection2!(g::RPCFGrid{S, DM, DEALIAS, PAD, PLAN, IPLAN}) where {S, DM, DEALIAS, PAD, PLAN, IPLAN} = new{S, DM, DEALIAS, PAD, PLAN, IPLAN}(RPCFField(g))
end
function (f::Convection2!{S})(∇uv::VectorField{3, RPCFField{S}}, u::VectorField{3, RPCFField{S}}, v::VectorField{3, RPCFField{S}}) where {S}
    ∇uv[1] .= 0.0
    mult_add!(∇uv[2], v[1], ddy!(f.cache[1], u[1]))
    mult_add!(∇uv[2], v[2], ddy!(f.cache[1], u[2]))
    mult_add!(∇uv[2], v[3], ddy!(f.cache[1], u[3]))
    mult_add!(∇uv[3], v[1], ddz!(f.cache[1], u[1]))
    mult_add!(∇uv[3], v[2], ddz!(f.cache[1], u[2]))
    mult_add!(∇uv[3], v[3], ddz!(f.cache[1], u[3]))
    return ∇uv
end
