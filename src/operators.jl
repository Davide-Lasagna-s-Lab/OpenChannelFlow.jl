# Definitions of the RHS operators for RPCF.

struct RPCFNavierStokesOperator{NSO}
    NSOper::NSO
    Ro::Float64

    function RPCFNavierStokesOperator(g::RPCFGrid, Re, Ro)
        nso = NavierStokesOperator(g, Re)
        new{typeof(nso)}(nso, Float64.(Ro))
    end
end

function (f::RPCFNavierStokesOperator)(N_u::VectorField{3}, u::VectorField{3})
    # compute standard NS operator
    f.NSOper(N_u, u)

    # compute cross product term
    @. N_u[1] += f.Ro*u[2]
    @. N_u[2] -= f.Ro*u[1]

    return N_u
end


struct RPCFGradientOperator{GRAD}
    GradOper::GRAD
    Ro::Float64

    function RPCFGradientOperator(g::RPCFGrid, Re, Ro)
        grad = GradientOperator(g, Re)
        new{typeof(grad)}(grad, Float64.(Ro))
    end
end

function (f::RPCFGradientOperator)(M_ur::VectorField{3}, u::VectorField{3}, r::VectorField{3})
    # compute standard gradient operator
    f.GradOper(M_ur, u, r)

    # compute cross product terms
    @. M_ur[1] += f.Ro*r[2]
    @. M_ur[2] -= f.Ro*r[1]

    return M_ur
end

Objective(g::RPCFGrid, Re::Real, Ro::Real, modes::Array{ComplexF64, 4}, base::Vector{Float64}) = ReSolverInterface.Objective(g, Re, modes, base, RPCFNavierStokesOperator(g, Re, Ro), RPCFGradientOperator(g, Re, Ro))
