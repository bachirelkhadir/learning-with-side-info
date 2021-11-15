ℓ₂(u) = begin
    u = reduce(vcat, u)
    sqrt(sum(u.^2) / size(u,1))
end
    
function get_trajectory(vf, x0, noise_level, length, Tf; verbose=true)
    Random.seed!(0)
    training_x = hcat(solve_ode(vf, x0; verbose=verbose, length=length, tspan=(0, Tf))...)'
    training_xdot = vf.([training_x[i, :] for i in 1:size(x0, 1)]...)
    training_xdot = hcat(training_xdot...)
    training_xdot = training_xdot + noise_level * randn(size(training_xdot))
    training_x, training_xdot
end

function get_trajectories(vf, initial_conditions, noise_level, length, Tf; verbose=true,)
    training_x = []
    training_xdot = []
    for x0 = initial_conditions
        training_x_i, training_xdot_i = get_trajectory(vf, x0, noise_level,
            length, Tf; verbose=verbose)
        push!(training_x, training_x_i)
        push!(training_xdot, training_xdot_i)
    end
    hcat(training_x...), hcat(training_xdot...)
end


∞ = 1e9
function trajectory_error(vf, true_vf, initial_conditions, num_samples, Tf; verbose=false)
    "Compute the error between vf and true_bf along trajectories starting from initial_conditions."
    try
        vf_at_x = get_trajectories(vf, initial_conditions, 0, num_samples, Tf, verbose=verbose)[2]
        true_vf_at_x = get_trajectories(true_vf, initial_conditions, 0, num_samples, Tf, verbose=verbose)[2]
        ℓ₂(vf_at_x .- true_vf_at_x)
    catch err
     ∞
    end

end

function box_error(vf, true_vf, box; verbose=false)
    "Compute the error between vf and true_bf on a discrete grid of box"
    xs = [
        range(b1, b2, length=Int(ceil(b2-b1)/dx)) for (b1, b2, dx)=box
    ]
    x = collect(Iterators.product(xs...))
    # adapt argument format for vf
    tilde = f -> (u -> f(u...))
    #mean(u) = sum(u) ./ size(u, 1)
    ℓ₂(tilde(vf).(x) .- tilde(true_vf).(x))
end

train_model(deg, training_x, training_xdot, side_info; verbose=false) = begin
    p_opt, opt_value = 
    fit_polynomial_to_data_with_side_info(training_x, training_xdot; 
        side_info=side_info, verbose=verbose, deg=deg, regularization=0.)
    p_opt_as_f = (a...) ->  map(pi -> pi(a), p_opt)
    p_opt, p_opt_as_f, opt_value
end

train_and_test_model(deg, noise_level, num_trajectories, num_samples, side_info,
    training_initial_conditions, test_initial_conditions, Tf, box; verbose=false) = begin
    training_x, training_xdot = get_trajectories(ff, training_initial_conditions[1:num_trajectories],
            noise_level, num_samples, Tf;
            verbose=verbose, )
    learned_vf_as_poly, learned_vf, opt_value = train_model(deg, training_x, training_xdot, side_info; verbose)
    Dict(
        "side_info" => collect(keys(side_info)),
        "deg" => deg,
        "noise_level" => noise_level, 
        "num_trajectories" => num_trajectories, 
        "num_samples" =>  num_samples, 
        "box"=>box_error(learned_vf, ff, box),
        "test"=> trajectory_error(learned_vf, ff, test_initial_conditions, num_samples, Tf),
        "training"=> trajectory_error(learned_vf, ff, training_initial_conditions[1:num_trajectories], num_samples, Tf),
        "learned_p" => learned_vf_as_poly, #string(learned_vf_as_poly),
        "obj_value" => opt_value / sqrt(prod(size(training_x))), # normalize
        "training_x" => training_x,
        "training_xdot" => training_xdot,
        
    )
end

