using DifferentialEquations

println("Loaded diff equations")


struct OdeDiverges <: Exception end
function solve_ode(vf, x0; verbose=true, tspan = (0., 30.0), length=10)
    if verbose
        println("Solving ODE from x0=$x0")
    end
    function vf_for_solver(dx, x, params, t)
        fx = vf(x...)
        for i in 1:size(x0,1)
            dx[i] = fx[i]
        end
    end
    prob = ODEProblem(vf_for_solver, x0, tspan, )
    alg = Euler()
    maxiters = 1e4
    sol = DifferentialEquations.solve(prob, alg, dt=(tspan[2]-tspan[1])/maxiters)
    trange = range(tspan[1], stop=tspan[2], length=length)
    if verbose
        @show size(sol)
        @show length
    end
    if size(sol, 2) <= 1
        throw(OdeDiverges())
    end
    sol = hcat(sol.(trange)...)
    return [sol[i, :] for i in 1:size(sol, 1)]
end


# function solve_ode_3d(vf, x0; verbose=true, tspan = (0., 30.0), length=10)
#     if verbose
#         println("Solving ODE from x0=$x0")
#     end
#     function vf_for_solver(dx, x, params, t)
#         dx[1], dx[2], dx[3] = vf(x[1], x[2], x[3])
#     end
#     prob = ODEProblem(vf_for_solver, x0, tspan)
#     sol = DifferentialEquations.solve(prob)
#     trange = range(tspan[1], stop=tspan[2], length=length)
#     if verbose
#         @show size(sol)
#         @show length
#     end
#     if size(sol, 2) <= 1
#         throw(OdeDiverges())
#     end
#     sol = hcat(sol.(trange)...)
#     return (sol[1, :], sol[2, :], sol[3, :])
# end
