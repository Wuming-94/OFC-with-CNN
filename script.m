%% Workspace initialization
clear all;
close all;
clc;

%% Parameters
% Medium refractive index
nm = 1.33; % Medium refractive index
np = 1.50; % Particle refractive index

% Focusing
f = 10e-6; % Focal length [m]
NA = 1.30; % numerical aperture
L = f*NA/nm; % Iris aperture [m]

% Trapping beam
Ex0 = 1e+4; % x electric field [V/m]
Ey0 = 1i*1e+4; % y electric field [V/m]
w0 = 100e-6; % Beam waist [m]
Nphi = 40; % Azimuthal divisions
Nr = 40; % Radial divisions
power = 5e-3; % Power [W]
resolution = 300;

%% Trapping beam Initialization

bg = BeamGauss(Ex0,Ey0,w0,L,Nphi,Nr);
bg = bg.normalize(power); % Set the power
% Calculates set of rays corresponding to optical beam
r = Ray.beam2focused(bg,f);

%Ask user for shape
shape = lower(input('Enter shape name (e.g. "spherical"): ', 's'));
N = input('Enter number of samples to generate:');

% Projection image coordinates
xv = linspace (-15,15, resolution);
yv = linspace (-15,15, resolution);
zv = linspace (-15,15, resolution);

% Create output directory
output_dir = shape;

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% % Write to file
% filepath = fullfile(output_dir, [shape, '_op.txt']);
% fid1 = fopen(filepath, 'a');
% parafile = fullfile(output_dir, [shape, '_para.txt']);
% fid2 = fopen(parafile, 'a');  % 'a' 表示追加写入

%% Iteration
for i = 1:N
    
    % Switch for shape type
    switch shape
        case 'spherical'
        % Random radius and position
        % R = (5 + rand()*5) * 1e-6; % Radius: 5–10 µm
            R = 5e-6;
            range_min = -15e-6 + R;
            range_max =  15e-6 - R;
            c = Point(range_min + rand() * (range_max - range_min), ...
                      range_min + rand() * (range_max - range_min), ...
                      range_min + rand() * (range_max - range_min));% Range: inside [-15,15]
            % Create spherical particle

            bead = ParticleSpherical(c, R, nm, np);

            % Projection
            [proj_xy, proj_yz, proj_xz] = spVP(bead, resolution);

            % fprintf(fid2, '%d\t%.6e\t%.6e\t%.6e\t%.2e\t%.3f\t%.3f\n', i, c.X, c.Y, c.Z, R, nm, np);

        case 'pyramid'
        % Random radius and position
        % R = (5 + rand()*5) * 1e-6; % Radius: 5–10 µm
            a = 4e-6;
            h = a;
            % V = [];
            % 
            % while isempty(V) || norm(V) <= 2.5
            %     V = rand(1,3);  % or -5 + 10*rand(1,3)
            % end
            vec = randn(1,3);        % 从正态分布中随机生成一个向量
            I = vec / norm(vec);       % 单位化
            V = h * I*3/4;             % 放缩到长度为 r

            range_min = -15e-6 + h*3/4;
            range_max =  15e-6 - h*3/4;

            X = range_min + rand() * (range_max - range_min); 
            Y = range_min + rand() * (range_max - range_min);
            Z = range_min + rand() * (range_max - range_min); % Range: inside [-15,15]
            % 
            % v = Vector(2.5e-6,0,0,0,0,3e-6);
            v = Vector(X,Y,Z,V(1),V(2),V(3));
            
            psi = 2*pi*rand();
            % psi = pi/6;
            % Create spherical particle
            bead = ParticlePyramid(v, a,psi, nm, np);

            % Projection
            [proj_xy, proj_yz, proj_xz] = pyVP2(bead, resolution);

            % fprintf(fid2, ...
            %     '%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.3f\t%.3f\n', ...
            %     i, v.X, v.Y, v.Z, v.Vx, v.Vy, v.Vz, a, psi, nm, np);

        case 'cube'
            d = 5e-6;  %cwidth
            vec = randn(1,3);        % 从正态分布中随机生成一个向量
            I = vec / norm(vec);       % 单位化
            V = d * I/2;             % Scaling to  d/2
            l = sqrt(3)*d/2; %length to corner

            range_min = -15e-6 + l; %
            range_max =  15e-6 - l;

            X = range_min + rand() * (range_max - range_min); 
            Y = range_min + rand() * (range_max - range_min);
            Z = range_min + rand() * (range_max - range_min); % Range: inside [-15,15]

            % v = [X,Y,Z,Vx,Vy,Vz];
            v = Vector(X,Y,Z,V(1),V(2),V(3));
            % v= Vector(5e-6,0,0,0,0,2.5e-6);

            psi = 2*pi*rand();
            % psi= 0;
            % Create spherical particle
            bead = ParticleCube(v,psi, nm, np);

            % Projection
           [proj_xy, proj_yz, proj_xz] = cuVP(bead, resolution);

            % fprintf(fid2, ...
            %     '%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.3f\t%.3f\n', ...
            %     i, v.X, v.Y, v.Z, v.Vx, v.Vy, v.Vz, psi, nm, np);
       
        otherwise
            error('Shape "%s" not implemented.', shape);
    end
    
    % Calculate force & torque
        forces  = bead.force(r)*1e15; % F: fN
        
        T = bead.torque(r)* 1e21; % T : fN·μm
        
        %Output 
        Op = zeros(1,6);
        Op = [sum(forces.Vx(isfinite(forces.Vx))), ...  
            sum(forces.Vy(isfinite(forces.Vy))), ...
            sum(forces.Vz(isfinite(forces.Vz))),...
            sum(T.Vx(isfinite(T.Vx))),...   
            sum(T.Vy(isfinite(T.Vy))),...
            sum(T.Vz(isfinite(T.Vz)))]; % isinfinite check NaN 1

  
      
        % % close file
        % fprintf(fid1, '%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', i, Op);
        % %data = readmatrix(fullfile('spherical', 'spherical_op.txt'));
        % 
        % % save images
        % % Plot XY ---
        % fig = figure('Visible','off');
        % imagesc(xv, yv, proj_xy');
        % colormap(parula); 
        % axis off; axis equal tight;
        % exportgraphics(fig, fullfile(output_dir, sprintf('Proj%d_xy.png', i)));
        % close(fig);
        % 
        % % Plot XZ ---
        % fig = figure('Visible','off');
        % imagesc(xv, zv, proj_xz');
        % colormap(parula);
        % axis off; axis equal tight;
        % exportgraphics(fig, fullfile(output_dir, sprintf('Proj%d_xz.png', i)));
        % close(fig);
        % 
        % % Plot YZ ---
        % fig = figure('Visible','off');
        % imagesc(yv, zv, proj_yz');
        % colormap(parula); 
        % axis off; axis equal tight;
        % exportgraphics(fig, fullfile(output_dir, sprintf('Proj%d_yz.png', i)));
        % close(fig);
end
    figure;
    forces.plot('scale', [1e+6 0.75e+9], ...
        'color', [0 0 0], ...
        'LineWidth', 2 ...
        );

     xlabel('X'); ylabel('Y'); zlabel('Z');
    
% fclose(fid1);
% fclose(fid2);





