# method definitions for vector calculus and derivative operations

# TODO: finish implementing vector calculus stuff

function ReSolverInterface.grad!(∇u::VectorField{3, RPCFField{S}}, u::RPCFField{S}) where {S} end
function ReSolverInterface.divergence!(div_u::RPCFField{S}, u::VectorField{3, RPCFField{S}}) where {S} end
function ReSolverInterface.laplacian!(Δu::RPCFField{S}, u::RPCFField{S}) where {S} end
function ReSolverInterface.convection!(u∇v::VectorField{3, RPCFField{S}}, u::VectorField{3, RPCFField{S}}, v::VectorField{3, RPCFField{S}}) where {S} end
function ReSolverInterface.convection2!(∇uv::VectorField{3, RPCFField{S}}, u::VectorField{3, RPCFField{S}}, v::VectorField{3, RPCFField{S}}) where {S} end
