using JuMP
using SumOfSquares
using MosekTools
using LinearAlgebra
using DynamicPolynomials



function fit_polynomial_to_data_with_side_info(training_x, training_xdot;
                                deg=2,
                                solver=nothing,
                                side_info = [],
                                verbose = true)
    n = size(training_x, 1)
    @polyvar x[1:n]

    if solver == nothing
        solver = with_optimizer(Mosek.Optimizer, QUIET=true)
    end
    model = SOSModel(solver)

    if verbose
        @show n
        @show size(training_xdot)
        @show size(training_x)
        @show deg
        @show solver
    end

    # define polynomial variable
    if verbose
        println("Define polynomial variable")
    end
    mon_x = monomials(x, 0:deg)
    @variable model p[1:n] Poly(mon_x)

    if verbose
        println("Define p(training)")
    end

    p_at_x = map(u -> [pi(x=>u) for pi in p],
                    collect(eachcol(training_x)))
    p_at_x = hcat(p_at_x...)


    # side information
    if verbose
        @show side_info
    end

    for (side_info_name, side_info_func) in side_info
        if verbose
            println("Imposing $(side_info_name)")
        end
        side_info_func(model, p, x)
    end


    # objective
    if verbose
        println("Define objective")
    end

    @variable model least_squares_error
    #@constraint model sum((p_at_x - training_xdot).^2) <= least_squares_error
    @constraint model [least_squares_error, (p_at_x - training_xdot)...] in SecondOrderCone()

    objective = least_squares_error

    # if verbose
    #     println("(maybe) Add regularization")
    # end

    # if regularization
    #     @variable model l2_penalty
    #     coeffs = coefficients.(p)
    #     coeffs = hcat(coeffs...)
    #     @constraint model norm(coeffs) <= l2_penalty
    #     objective = objective + regularization_scaling * l2_penalty
    # end

    @objective model Min objective


    # solve model
    if verbose
        println("Solving")
    end

    optimize!(model)

    # retirm solution
    p_least_squares = value.(p)

    if verbose
        @show value(objective)
    end
    #p_as_f = u -> map(pi -> pi(x=>u), p_least_squares)

    return p_least_squares, value(objective)
end
