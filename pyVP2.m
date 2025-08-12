function [proj_xy, proj_yz, proj_xz] = pyVP2(obj, resolution)
% PYRAMIDVOLUMEPROJECTION Compute 3D volume projections of a square pyramid
%
% Inputs obj:
%   v   - center orientation vectors (Vector) 
%   a        - base edge length
%   psi    - rotation angle around dir (in radians)
%   resolution - number of voxels per dimension (default: 200)
%
% Outputs:
%   proj_xy  - XY projection (2D matrix)
%   proj_yz  - YZ projection (2D matrix)
%   proj_xz  - XZ projection (2D matrix)

if nargin < 6
    resolution = 200;
end

% Convert to micrometers
    Scale = 1e6;
    % margin = 1;
    a = obj.pyr.a * Scale;
    center = [obj.pyr.v.X,obj.pyr.v.Y,obj.pyr.v.Z].* Scale; %[μm]
    dir = [obj.pyr.v.Vx,obj.pyr.v.Vy,obj.pyr.v.Vz].* Scale; %1x3
    dir_unit = dir / norm(dir); %3x1

    h = 4*norm(dir)/3; % length
    weights = real(obj.np - obj.nm); % Refractive index contrast

    % 1. Local coordinates of the pyramid
    n_local = [0;0;1];             % Local z-axis

    % 2. Compute rotation matrix
   
    vec = cross(n_local, dir_unit);
    s = norm(vec);
    if s < 1e-8
        R_align = eye(3);
    else
        c = dot(n_local, dir_unit);

        vx = [   0   -vec(3)  vec(2);
               vec(3)    0  -vec(1);
              -vec(2)  vec(1)    0 ];

        R_align = eye(3) + vx + vx^2 * ((1-c)/s^2);
    end
    
    K = [   0           -dir_unit(3)  dir_unit(2);
          dir_unit(3)    0           -dir_unit(1);
         -dir_unit(2)  dir_unit(1)    0 ];

    R_theta = eye(3) + sin(obj.pyr.psi)*K + (1 - cos(obj.pyr.psi))*K^2;
    
    R =  R_theta*R_align;
    
    % 3. meshgrid
    xv = linspace (-15,15, resolution);
    yv = linspace (-15,15, resolution);
    zv = linspace (-15,15, resolution);
    
    [X, Y, Z] = ndgrid(xv,yv,zv);
    
    % 4. Transform  points into local coordinates
    R_inv = R';
    pts = [X(:)'; Y(:)'; Z(:)'];
    pts_local = R_inv * (pts - center');
    
    x_l = pts_local(1,:);
    y_l = pts_local(2,:);
    z_l = pts_local(3,:);
    
    % 5. Inside-pyramid condition
    in_z = (z_l >= -h/4) & (z_l <= 3*h/4);        % Total height is h        % Within height
    z_rel = z_l + h/4;                            % z_rel ∈ [0, h]
    a_h = a * (1 - z_rel / h);
    a_h(z_rel < 0 | z_rel > h) = 0;
    
    half_a_h = a_h / 2;
    in_xy = (abs(x_l) <= half_a_h) & (abs(y_l) <= half_a_h);
    
    inside = in_z & in_xy;


    % 6. Reshape mask
    mask = reshape(inside, size(X));
    
    % 9. Compute projections
    proj_xy = squeeze(sum(mask.* weights, 3));
    proj_yz = squeeze(sum(mask.* weights, 1));
    proj_xz = squeeze(sum(mask.* weights, 2));

% % 7. Display projections and save to images
% U = linspace(-15, 15, resolution);
% V = linspace(-15, 15, resolution);
% 
% % 创建保存路径
% output_dir = 'pyramid';
% if ~exist(output_dir, 'dir')
%     mkdir(output_dir);
% end
% 
% % 创建每张图像并保存（不显示，只保存文件）
% % --- XY ---
% fig = figure('Visible','off');
% imagesc(xv, yv, proj_xy');
% colormap(parula); 
% axis off; axis equal tight;
% exportgraphics(fig, fullfile(output_dir, 'Proj1_xy.png'));
% close(fig);
% 
% 
% % --- XZ ---
% fig = figure('Visible','off');
% imagesc(xv, zv, proj_xz');
% colormap(parula);
% axis off; axis equal tight;
% exportgraphics(fig, fullfile(output_dir, 'Proj1_xz.png'));
% close(fig);
% 
% % --- YZ ---
% fig = figure('Visible','off');
% imagesc(yv, zv, proj_yz');
% colormap(parula); 
% axis off; axis equal tight;
% exportgraphics(fig, fullfile(output_dir, 'Proj1_yz.png'));
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
