# Definitions of the RHS operators for RPCF.

struct RPCFNavierStokesOperator{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    NSOper::NavierStokesOperator{3, RPCFField{S, DM, DEALIAS, PAD, PLAN, IPLAN}, Convergence!, Laplacian!}
    Ro::Float64

    RPCFNavierStokesOperator(g::RPCFGrid{S, DM, DEALIAS, PAD, PLAN, IPLAN}, Re::Float64, Ro::Float64) where {S, DM, DEALIAS, PAD, PLAN, IPLAN} = new{S, DM, DEALIAS, PAD, PLAN, IPLAN}(NavierStokesOperator(RPCFField, g, 3, Re, conv=Convection!(g), lapl=Laplacian!(g)), Ro)
end

function (f::RPCFNavierStokesOperator)(N_u::VectorField{3}, u::VectorField{3})
    # compute standard NS operator
    f.NSOper(N_u, u)

    # compute cross product term
    @. N_u[1] -= f.Ro*u[2]
    @. N_u[2] += f.Ro*u[1]

    return N_u
end


struct RPCFGradientOperator{S, DM, DEALIAS, PAD, PLAN, IPLAN}
    GradOper::GradientOperator{3, RPCFField{S, DM, DEALIAS, PAD, PLAN, IPLAN}, Convection!, Convection2!, Laplacian!}
    Ro::Float64

    RPCFGradientOperator(g::RPCFGrid{S, DM, DEALIAS, PAD, PLAN, IPLAN}, Re::Float64, Ro::Float64) where {S, DM, DEALIAS, PAD, PLAN, IPLAN} = new{S, DM, DEALIAS, PAD, PLAN, IPLAN}(GradientOperator(RPCFField, g, 3, Re, conv1=Convection!(g), conv2=Convection2!(g), lapl=Laplacian!(g)), Ro)
end

function (f::RPCFGradientOperator)(M_ur::VectorField{3}, u::VectorField{3}, r::VectorField{3})
    # compute standard gradient operator
    f.GradOper(M_ur, u, r)

    # compute cross product terms
    @. M_ur[1] += f.Ro*u[2]
    @. M_ur[2] -= f.Ro*u[1]

    return M_ur
end

# TODO: test this
ReSolverInterface.Objective(g::RPCFGrid, Re::Float64, Ro::Float64, modes::Array{ComplexF64, 4}, base::Vector{Float64}, free_mean::Bool=true) = ReSolverInterface.Objective(RPCFField, g, 3, Re, modes, base, free_mean, RPCFNavierStokesOperator(g, Re, Ro), RPCFGradientOperator(g, Re, Ro))
