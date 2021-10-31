%% load libraries
addpath(genpath(['~/yalmip/yalmip/YALMIP-master']))
addpath(genpath('~/mosek/mosek'))
ops = sdpsettings('solver','mosek','verbose',0);

%%
% vf = @(phi, w) [w; sin(phi)];

syms phi(t);
[V] = odeToVectorField(diff(phi, 2) == -sin(phi))
vf = matlabFunction(V);
vf_t = matlabFunction(V, 'vars', {'t', 'Y'});

%% compute and plot trajectories
start_phi = -1:.1:1;
start_w = -ones(size(start_phi));

sample_trajectory = []
t_limits = [0, 2*pi];
ts = t_limits(1):.1:t_limits(2);
for i=1:5
    sol = ode45(vf_t, t_limits, [start_phi(i), start_w(i)]);
    sol_as_function = @(t) deval(sol, t, 1);
    sample_trajectory = [sample_trajectory; sol_as_function(ts)];
    %fplot(@(t) deval(sol, t, 1), [0, 2*pi])
end

figure(1)
clf
hold on
plot(ts, sample_trajectory);


%% Fit a vectorfield using least squares

deg_p = 4; % degree of p
d = 1; % dimension
x = sdpvar(d, 1);
p = sdpvar(d, 1);
cp = [];
for i=1:d
    [pi, cpi] = polynomial(x, deg_p);
    p(i) = pi;
    cp = [cp; cpi'];
end

% sum_i int ||p(x_i(t)) - x_dot_i(t)||
p_x = sdpvar(size(sample_trajectory, 1), ...
             size(sample_trajectory, 2));
x_dot = sdpvar(size(sample_trajectory, 1), ...
             size(sample_trajectory, 2));
for i=1:size(sample_trajectory, 1)
    for j=1:size(sample_trajectory, 2)
        p_x(i, j) = replace(p, x, sample_trajectory(i, j));
        x_dot(i, j) = -sin(sample_trajectory(i, j));
    end
end
least_squares_error = p_x - x_dot;
objective = norm(least_squares_error, 2);

sol = optimize([], objective, ops)
p_sol = replace(p, cp, value(cp))

f = sdisplay(p_sol);
f = f{1};
f = replace(f,'*','.*');
f = replace(f,'^','.^');
f = eval(['@(x)' f]);
x = -2:0.01:2;
figure(2)
clf
hold on
plot(x, f(x));
plot(x, -sin(x));


%% plot vf
figure(2)
clf
hold on
xlabel('phi')
ylabel('phidot')

flatten = @(u) reshape(u, 1, prod(size(u)));
[phi,w] = meshgrid(-pi:0.5:pi,-pi:0.5:pi);
phi = flatten(phi);
w = flatten(w);
phi_dot = [];
w_dot = [];
for i=1:prod(size(phi))
    dy = vf([phi(i); w(i)]);
    phi_dot(i) = dy(1);
    w_dot(i) = dy(2);
end
quiver(phi, w, phi_dot, w_dot)


T = 10;
x0 = [-2, 2];
sol = ode45(vf_t, [0, T], x0);
y = deval(sol, 0:0.01:T, [1, 2]);
plot(y(1, :), y(2, :), 'r', 'linewidth', 3)

%% vector field
[phi,w] = meshgrid(-1:0.1:1,-1:0.1:1);
phi_dot = w;
w_dot = -sin(phi);


%% plot
start_phi = -1:5j:1;
start_w = -ones(size(start_phi))*0;

figure(1)
clf
quiver(phi, w, phi_dot, w_dot)
streamline(phi,w,phi_dot,w_dot,start_phi,start_w)