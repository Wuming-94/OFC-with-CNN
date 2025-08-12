function [proj_xy, proj_yz, proj_xz] = spVP(obj, resolution)
% PYRAMIDVOLUMEPROJECTION Compute 3D volume projections of a square pyramid
%
% Inputs:
%   c   - 3x1 base center position
%   r        - radien
%   resolution - number of voxels per dimension (default: 200)
%
% Outputs:
%   proj_xy  - XY projection (2D matrix)
%   proj_yz  - YZ projection (2D matrix)
%   proj_xz  - XZ projection (2D matrix)

if nargin < 3
    resolution = 200;
end

% Convert to micrometers
    Scale = 1e6;
    % margin = 1;

% 1. Local coordinates of the pyramid
    rad = obj.sp.r * Scale;
    pos = [obj.sp.c.X, obj.sp.c.Y, obj.sp.c.Z ].* Scale;
    weights = real(obj.np - obj.nm); % Refractive index contrast
    
% 2. meshgrid
    xv = linspace (-15,15, resolution);
    yv = linspace (-15,15, resolution);
    zv = linspace (-15,15, resolution);

    [X, Y, Z] = ndgrid(xv,yv,zv);
    
    % Initialize projection matrices
    proj_xy = zeros(resolution);
    proj_xz = zeros(resolution);
    proj_yz = zeros(resolution);

     % Sphere mask calculation
    mask = (X - pos(:,1)).^2 + (Y - pos(:,2)).^2 +(Z - pos(:,3)).^2 <= rad^2;


  % Accumulate projections
    proj_xy = squeeze(sum(mask .* weights, 3));
    proj_xz = squeeze(sum(mask .* weights, 2));
    proj_yz = squeeze(sum(mask .* weights, 1));

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
