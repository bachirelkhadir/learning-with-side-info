using DifferentialEquations

println("Loaded diff equations")


struct OdeDiverges <: Exception end
function solve_ode(vf, x0; verbose=true, tspan = (0., 30.0), length=10)
    if verbose
        println("Solving ODE from x0=$x0")
    end
    function vf_for_solver(dx, x, params, t)
        dx[1], dx[2] = vf(x[1], x[2])
    end
    prob = ODEProblem(vf_for_solver, x0, tspan)
    sol = DifferentialEquations.solve(prob)
    trange = range(tspan[1], stop=tspan[2], length=length)
    if verbose
        @show size(sol)
        @show length
    end
    if size(sol, 2) <= 1
        throw(OdeDiverges())
    end
    sol = hcat(sol.(trange)...)
    return (sol[1, :], sol[2, :])
end

