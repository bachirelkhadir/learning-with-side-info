using JuMP
using SumOfSquares
using MosekTools
using LinearAlgebra
using DynamicPolynomials

 

function fit_polynomial_to_data(training_x, training_xdot;
                                deg=2,
                                solver=nothing, 
                                box_invariance=false, 
                                monotonicity=false,
                                equilibrium_points=false,
                                regularization = false,
                                regularization_scaling = 1e-3,
                                verbose = true)
    @polyvar x y

    if solver == nothing
        solver = with_optimizer(Mosek.Optimizer, QUIET=true)
    end
    model = SOSModel(solver)

    if verbose
        @show size(training_xdot)
        @show size(training_x)
        @show deg
        @show solver
        @show model
        @show box_invariance
        @show monotonicity
        @show equilibrium_points
        @show regularization
    end

    # define polynomial variable
    if verbose
        @show "Define polynomial variable"
    end
    mon_xy = monomials((x, y), 0:deg)
    @variable model p1 Poly(mon_xy)
    @variable model p2 Poly(mon_xy)
    p = [p1; p2]

    if verbose
        @show "Define p(training)"
    end
    p_at_x = [map((u1,u2) -> pi(x=>u1, y=>u2),
                training_x[1, :], training_x[2, :])
              for pi in [p1, p2]]
    p_at_x = hcat(p_at_x...)'

    if verbose
        @show "Side info"
    end
    # side information
    if box_invariance
        int_y = @set y >= 0 && 1 - y >= 0
        @constraint(model, subs(p1, x=>0) >= 0, domain=int_y )
        @constraint(model, -subs(p1, x=>1) >= 0, domain=int_y )

        int_x = @set x >= 0 && 1 - x >= 0
        @constraint(model, subs(p2, y=>0) >= 0, domain=int_x )
        @constraint(model, -subs(p2, y=>1) >= 0, domain=int_x )
    end

    if monotonicity
        Box = @set 1 - x >=  0 && x >= 0 && 1 - y >= 0 && y >= 0
        @constraint(model, differentiate(p1, y) >= 0, domain=Box)
        @constraint(model, differentiate(p2, x) >= 0, domain=Box)
    end
    if equilibrium_points
        @constraint(model, subs(p1, x=>0, y=>0) == 0)
        @constraint(model, subs(p2, x=>0, y=>0) == 0)
    end

    if verbose
        @show "Define objective"
    end

        
    # objective
    @variable model least_squares_error 
    #@constraint model sum((p_at_x - training_xdot).^2) <= least_squares_error
    @constraint model [least_squares_error, (p_at_x - training_xdot)...] in SecondOrderCone()

    objective = least_squares_error
    
    if verbose
        @show "(maybe) Add regularization"
    end
    
    if regularization        
        @variable model l2_penalty 
        coeffs = coefficients.(p)
        coeffs = hcat(coeffs...)
        @constraint model norm(coeffs) <= l2_penalty        
        objective = objective + regularization_scaling * l2_penalty
    end
    
    @objective model Min objective
    if verbose
        @show "Solving"
    end
    # solve model
    optimize!(model)

    p_least_squares = value.(p)
    if verbose
        @show value(objective)
    end
    p_as_f = (a, b) -> map(pi -> pi(x=>a, y=>b), p_least_squares)
    
    return p_as_f, x, y, p_least_squares, value(objective)
end
