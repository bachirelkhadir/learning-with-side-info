using Pkg
pkg"activate learning_ds"

include("learn_polynomial_with_side_info.jl")
include("plotting_utilities.jl")
include("ode_helper.jl")
using CSV
using DataFrames
using DataFramesMeta
using Dates
using TimeSeries
using HTTP
using DynamicPolynomials

## Matplotlib configuration
PyPlot.rc("font", family = "sans-serif", size = 10)
PyPlot.rc("figure", autolayout = true)

## SIR sample example

γ, β = .1, .5
SIR_vf(S, I) = [-β * I * S, β * I * S - γ * I]

# plot SIR sample example
PyPlot.figure(1)
plot_vectorfield(SIR_vf, (-0.01, 1.01, 1000), (-0.01, 1.01, 1000))
PyPlot.xlabel("\$S\$")
PyPlot.ylabel("\$I\$")
PyPlot.xlim(-.1, 1.1)
PyPlot.ylim(-.1, 1.1)
PyPlot.xticks([0, 1])
PyPlot.yticks([0, 1])
PyPlot.title("SIR model normalized")
plot_box()



## Download data from CSSEGISandData

println("Downloading data...")

base_url =
"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/817c2bd1f419d553c11a7d3e126827746b903dec/csse_covid_19_data/csse_covid_19_time_series/"

data_urls = Dict(
    :infected => base_url * "time_series_covid19_confirmed_global.csv",
    :recovered => base_url * "time_series_covid19_recovered_global.csv",
    :dead => base_url * 
    "time_series_covid19_deaths_global.csv"
)

data = Dict(
    data_name => 
    #CSV.read(IOBuffer(HTTP.get(data_url).body)) 
    readtable(IOBuffer(HTTP.get(data_url).body), )
    for (data_name, data_url) in data_urls)

@show DataFrames.first(data[:dead], 5)

println("Preparing data")

function sum_data(data)
    """
    Aggregate statistics across countries.
    """

    function symbol_to_date(symb)
        symb_split = split(string(symb)[2:end], "_")
        y = 2020
        m, d, y = parse.(Int32, symb_split)
        y += 2000
        Date(Int32(y), Int32(m), Int32(d))
    end
     
    cols = Dict()
    ts = 0
    for (df_name, df) in data
        dates = names(df)[5:end]
        col = [ sum(df[:, date]) for date in dates]
        ts = [symbol_to_date(date) for date in dates]
        push!(cols, df_name => col)
        
    end
    push!(cols, :index => ts)
    DataFrame(cols)
end


summed_data = sum_data(data)
@show DataFrames.first(summed_data, 5)


## Prepare training data
println("Preparing training data")

# x = (S, I)
# S + I + R = S0

N0 = 3e6
S0 = .9
I = summed_data[!, :infected]  ./ N0
R = (summed_data[!, :dead] + summed_data[!, :recovered] ) ./ N0
S = S0 .- R .- I

x = Float32.(hcat(S, I))' 
ẋ = x[:, 2:end] - x[:, 1:end-1]
x = x[:, 1:end-1]

training_x = x
training_xdot = ẋ
size(x), size(ẋ)
PyPlot.plot(x[1, :], x[2, :], color=:red, ls="--", lw=5, label="training data")


## Side info

# invariance simplex
inv_simplex(model, p, x) = 
    [[

        @constraint(model, subs(p[i], x[i]=>0) >= 0)
        for i=1:size(x,1)
    ]...,
     @constraint(model, subs(p[1]+p[2], x[1]=>1-x[2]) <= 0) ]


monotonicity(model, p, x) = begin
    S, I = x
    Ṡ, İ = p
    pos_orthand = @set x[1] >= 0 &&  x[2] >= 0 && (sum(x) <= 1)
    [ 
        
        @constraint(model, Ṡ <= 0, domain=pos_orthand),
        @constraint(model, differentiate(Ṡ, I) <= 0, domain=pos_orthand),
        @constraint(model, differentiate(Ṡ, S) <= 0, domain=pos_orthand),
        @constraint(model, differentiate(İ, S) >= 0, domain=pos_orthand)
    ]

end


eq_at_zero(model, p, x) = [ @constraint(model, coefficients(subs(pi,
    xi => 0)) .== 0) for (xi,pi)=zip(x,p) ]

side_info = Dict(
    "inv simplex" => inv_simplex,
    "equibilrium points" => eq_at_zero,
    "monotonicity" => monotonicity
)

d = 4

p_opt, opt_value = 
            fit_polynomial_to_data_with_side_info(training_x, training_xdot; 
    deg=d, verbose=false, side_info=side_info)
p_opt_as_f = (a, b) ->  map(pi -> pi([a,b]), p_opt)

@show opt_value
x = variables(p_opt)
@polyvar S I
round.([pi(x => [S, I]) for pi in p_opt], digits=2) 


## Plot solution
vf = p_opt_as_f
vf_as_poly = p_opt
mytitle = "Least squares fit with side information"

PyPlot.title(mytitle)
PyPlot.clf()
initial_conditions = [ training_x[:, 1], [1., 0.], [.3, .2]]    
for x0 in initial_conditions
    sol = solve_ode(vf, x0, tspan=(0., 100.), length=1000, verbose=false)
    PyPlot.plot(sol..., color=:black)
    PyPlot.scatter(x0[1], x0[2], color=:black, s=50)
end
PyPlot.plot(training_x[1, :], training_x[2, :], color=:red, ls="--", lw=5, label="training data")
PyPlot.title("Least squares fit with side information")

plot_vectorfield(vf, (-0.1, 1.1, 10), (-0.1, 1.1, 10))
PyPlot.xlim(-.1, 1.1)
PyPlot.ylim(-.1, 1.1)
PyPlot.xlabel("S")
PyPlot.ylabel("I");
PyPlot.axvline(0, color=:black, ls="--")
PyPlot.axhline(0, color=:black, ls="--")
PyPlot.plot([1, 0], [0, 1], color=:black, ls="--")

#PyPlot.savefig("../LatexDrafts/img/learned_vf_diffusion_disease_with_side_info.png")
 
