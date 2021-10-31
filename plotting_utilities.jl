using PyPlot
using DataFrames
using CSV

function plot_vectorfield(f, limits_x, limits_y)
    x = range(limits_x[1], stop=limits_x[2], length=limits_x[3])
    y = range(limits_y[1], stop=limits_y[2], length=limits_y[3])
    xx = broadcast((x, y) -> x, x', y)
    yy = broadcast((x, y) -> y, x', y)
    fXY = broadcast(f, x', y)
    fXY_X = map(u -> u[1], fXY)
    fXY_Y = map(u -> u[2], fXY)
    PyPlot.streamplot(xx, yy, fXY_X, fXY_Y, linewidth=1)
end

function plot_box()
    PyPlot.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], "--", color=:black)
end

function plot_univarite_poly(poly, var, limits=(-1, 1, 100); kwargs...)
    print(kwargs...)
    poly_as_func = ti -> poly(var => ti)
    t_range = collect(range(limits[1], stop=limits[2], length=limits[3]))
    p_t = map(poly_as_func, t_range)
    PyPlot.plot(t_range, p_t; kwargs...)
end


function plot_restriction_to_box(poly, x, y)
    limits = (-.1, 1.1, 100)
    for side in (0, 1)
        plot_univarite_poly(subs(poly[1], x=>side), y, limits, label="\$p_1($side, y)\$")
        plot_univarite_poly(subs(poly[2], y=>side), x, limits, label="\$p_2(x, $side)\$")
        plot_univarite_poly(0*x, x, limits, ls="--", color=:black)

    end
    PyPlot.legend()
end


function latex_vectorfield(f, limits_x, limits_y)
    my_streamplot = plot_vectorfield(f, limits_x, limits_y)
    transform = PyPlot.axes().transData.inverted().transform

    arrows = []
    names = split("x1 x2 x3 y1 y2 y3")
    names = Symbol.(names)
    for a in my_streamplot.arrows.get_paths()
        v = transform(a.to_polygons()[2])
        push!(arrows, [v[1:3, 1]..., v[1:3, 2]...])
    end 
    arrows = hcat(arrows...)'
    #DataFrame(columns=[arrows[i, :] for i in 1:6], names=names)
    arrows = convert(DataFrame, arrows)
    arrows = names!(arrows, names)


    segments = my_streamplot.lines.get_segments()
    segments = map(v -> vcat(v...), segments)
    segments = hcat(segments...)'
    segments = DataFrame(x1=segments[:,1], x2=segments[:,2],
                y1=segments[:,3], y2=segments[:,4])
    return my_streamplot, segments, arrows
end
