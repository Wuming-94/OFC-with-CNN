function [proj_xy, proj_yz, proj_xz] = cuVP(obj, resolution)
% CUBEVOLUMEPROJECTION Compute 3D volume projections of a cube
%
% Inputs:
%   obj.v    - 3x1 direction vector for cube orientation
%   obj.theta  - rotation around dir (in radians)
%   obj.np, obj.nm - refractive indices
%   resolution - number of voxels per dimension (default: 200)
%
% Outputs:
%   proj_xy  - XY projection (2D matrix)
%   proj_yz  - YZ projection (2D matrix)
%   proj_xz  - XZ projection (2D matrix)

if nargin < 2
    resolution = 200;
end

% === 1. Parameters ===
Scale = 1e6; % Convert to micrometers
L = 2 * obj.cb.v.norm() * Scale;
center = [obj.cb.v.X, obj.cb.v.Y, obj.cb.v.Z]* Scale;
theta = obj.cb.psi;
dir = [obj.cb.v.Vx, obj.cb.v.Vy, obj.cb.v.Vz]* Scale;
% weights = real(obj.np - obj.nm); % Refractive index contrast
weights = 1;
% === 2. Compute rotation matrix ===
n_local = [0; 0; 1]; % Local z-axis
dir_unit = dir / norm(dir);
v = cross(n_local, dir_unit);
s = norm(v);
if s < 1e-8
    R_align = eye(3);
else
    c = dot(n_local, dir_unit);
    vx = [0 -v(3) v(2);
        v(3) 0 -v(1);
        -v(2) v(1) 0];

    R_align = eye(3) + vx + vx^2 * ((1 - c) / s^2);
end

K = [0 -dir_unit(3) dir_unit(2); 
    dir_unit(3) 0 -dir_unit(1); 
    -dir_unit(2) dir_unit(1) 0];

R_theta = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;

R =  R_theta*R_align;

% === 3. Meshgrid in global coordinates ===
xv = linspace(-15, 15, resolution);
yv = linspace(-15, 15, resolution);
zv = linspace(-15, 15, resolution);
[X, Y, Z] = ndgrid(xv, yv, zv);

% === 4. Transform to local coordinates ===
R_inv = R';
pts = [X(:)'; Y(:)'; Z(:)'];
pts_local = R_inv * (pts - center');
% pts_local = R * pts + center';

x_l = pts_local(1,:);
y_l = pts_local(2,:);
z_l = pts_local(3,:);

% === 5. Inside cube condition ===
half_L = L / 2;
inside = (abs(x_l) <= half_L) & (abs(y_l) <= half_L) & (abs(z_l) <= half_L);

% === 6. Reshape mask ===
mask = reshape(inside, size(X));

% === 7. Compute projections ===
proj_xy = squeeze(sum(mask * weights, 3));
proj_yz = squeeze(sum(mask * weights, 1));
proj_xz = squeeze(sum(mask * weights, 2));

% % === 8. Save images ===
% output_dir = 'cube';
% if ~exist(output_dir, 'dir')
%     mkdir(output_dir);
% end
% 
% % --- XY ---
% fig = figure('Visible','off');
% imagesc(xv, yv, proj_xy');
% colormap(parula);
% axis off; axis equal tight;
% exportgraphics(fig, fullfile(output_dir, 'Proj_cube_xy.png'));
% close(fig);
% 
% % --- XZ ---
% fig = figure('Visible','off');
% imagesc(xv, zv, proj_xz');
% colormap(parula);
% axis off; axis equal tight;
% exportgraphics(fig, fullfile(output_dir, 'Proj_cube_xz.png'));
% close(fig);
% 
% % --- YZ ---
% fig = figure('Visible','off');
% imagesc(yv, zv, proj_yz');
% colormap(parula);
% axis off; axis equal tight;
% exportgraphics(fig, fullfile(output_dir, 'Proj_cube_yz.png'));
% close(fig);

figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1)
imagesc(xv, yv, proj_xy');
axis xy; axis equal tight
xlabel('x [μm]'); ylabel('y [μm]');
title('Projection XY'); colorbar;

subplot(1,3,2)
imagesc(xv, zv, proj_xz');
axis xy; axis equal tight
xlabel('x [μm]'); ylabel('z [μm]');
title('Projection XZ'); colorbar;

subplot(1,3,3)
imagesc(yv, zv, proj_yz');
axis xy; axis equal tight
xlabel('y [μm]'); ylabel('z [μm]');
title('Projection YZ'); colorbar;

end
