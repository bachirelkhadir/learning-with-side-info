function dy = pendulum_ode(t, y)
dy(1) = y(2);
dy(2) = y(1)*y(2)-2;