function [f, g] = loss_1ode(p, x, f, df_dy, y0)
n = length(p) / 3;
w = p(1:n);
v = p(n+1:2*n);
theta = p(2*n+1:3*n);
% Calculate objective f
z = x * w' - theta';
phi = 1 ./ (1 + exp(-z));
N = sum(v' .* phi, 2);
yt = y0 + x .* N;
ft = f(x, yt);
dphi = phi .* (1 - phi);
dN_dx = sum((v .* w)' .* dphi, 2);
dyt_dx = N + x .* dN_dx;
f = sum((dyt_dx - ft) .^ 2);
if nargout > 1 % gradient required
    dN_dp = [v' .* dphi .* x, phi, - v' .* dphi];
    ddphi = dphi .* (1 - 2 * phi);
    dN_dxdp = [v' .* dphi + (v .* w)' .* ddphi .* x, ...
        w' .* dphi, - (v .* w)' .* ddphi];
    dyt_dxdp = dN_dp + x .* dN_dxdp;
    dyt_dp = x .* dN_dp;
    dft_dp = df_dy(x, yt) .* dyt_dp;
    g = sum(2 * (dyt_dx - ft) .* (dyt_dxdp - dft_dp), 1)';
end

end