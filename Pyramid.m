classdef Pyramid < Superficies
    % Pyramid < Plane : Set of square pyramids in 3D
    % Each pyramid is defined by:
    %  v - center + dir [m]
    %  a-  base edge length a  [m]
    %  
    %  theta - a rotation angle psi around its axis

    properties
        v       % Vector (center and direction)
        a       % base edge length
        psi   % rotation angle (radians)
    end

    methods
        function obj = Pyramid(v,a,psi)
            Check.isa('v must be a Vector',v,'Vector')
            Check.isreal('a must be real > 0',a,'>',0)
        
            Check.isreal('psi must be real',psi)
            Check.samesize('v, a, psi must have same size',v,a,psi)

            obj.v = v;
            obj.a = a;
            obj.psi = psi;
        end

        function h = plot(pyr,varargin)

            N = 32;
            for n = 1:2:length(varargin)
                if strcmpi(varargin{n},'range')
                    N = varargin{n+1};
                end
            end
            
            % Scaling factor
            S = 1;
            for n = 1:2:length(varargin)
                if strcmpi(varargin{n},'scale')
                    S = varargin{n+1};
                    Check.isreal('The scaling factor must be a positive real number',S,'>',0)
                end
            end
            
            % Color level
            C = 0;
            for n = 1:2:length(varargin)
                if strcmpi(varargin{n},'colorlevel')
                    C = varargin{n+1};
                    Check.isreal('The scaling factor must be a real number',C)
                end
            end   

            Scale = 1e6;       
         
            center = [pyr.v.X,pyr.v.Y,pyr.v.Z].* Scale; %[μm]

            dir = [pyr.v.Vx,pyr.v.Vy,pyr.v.Vz].* Scale; %1x3

            h = 4*norm(dir)/3; % length
            top_local = [0; 0; h*3/4];

            half_a = pyr.a* Scale / 2;
            base_local = [...
               -half_a, -half_a, -h/4;
                half_a, -half_a, -h/4;
                half_a,  half_a, -h/4;
               -half_a,  half_a, -h/4]';

            [R_align,R_theta] = RMatrix(pry);
            
            R =  R_theta*R_align;

            base_world = R * base_local + center';
            top_world  = R * top_local + center';

            vertices = [base_world, top_world];  % 4底+1顶，总5个点
            vertices = vertices';

            faces = [
                1 2 3 4;  % 底面
                1 2 5 NaN;  % 侧面1
                2 3 5 NaN;  % 侧面2
                3 4 5 NaN;  % 侧面3
                4 1 5 NaN;  % 侧面4
            ];
            
            % === 5. 绘图 ===
            colors = [
                0.6 0.6 0.6;  % 底面灰
                1.0 0.2 0.2;  % 红
                0.2 1.0 0.2;  % 绿
                0.2 0.2 1.0;  % 蓝
                1.0 1.0 0.2;  % 黄
            ];
            
            hold on
            for i = 1:size(faces, 1)
                valid_idx = ~isnan(faces(i,:));
                h_out(i) = patch('Vertices', vertices, ...
                                 'Faces', faces(i,valid_idx), ...
                                 'FaceColor', colors(i,:), ...
                                 'FaceAlpha', 0.7, ...
                                 'EdgeColor', 'k');
            end
            hold off
            
            axis equal
            xlabel('X [μm]'); ylabel('Y[μm]'); zlabel('Z[μm]');
            grid on
            view(3)
        end

        function n = numel(pyr)
            n = numel(pyr.v);
        end

        function s = size(pyr,varargin)
            s = pyr.v.size(varargin{:});
        end

        function pyr_t = translate(pyr,dp)
            Check.isa('dp must be Point or Vector',dp,'Point','Vector')
            pyr_t = pyr;
            pyr_t.v = pyr.v.translate(dp);
        end

        function pyr_r = xrotation(pyr,phi)
            Check.isreal('phi must be real',phi)
            pyr_r = pyr;
            pyr_r.v = pyr.v.xrotation(phi);
        end

        function pyr_r = yrotation(pyr,phi)
            Check.isreal('phi must be real',phi)
            pyr_r = pyr;
            pyr_r.v = pyr.v.yrotation(phi);
        end

        function pyr_r = zrotation(pyr,phi)
            Check.isreal('phi must be real',phi)
            pyr_r = pyr;
            pyr_r.v = pyr.v.zrotation(phi);
        end

        function [R_align , R_theta] = RMatrix(pry)
         
            dir = [pry.v.Vx, pry.v.Vy, pry.v.Vz];
            theta = pry.psi;

            n_local = [0; 0; 1]; % Local z-axtis
            dir_unit = dir / norm(dir);

            vec = cross(n_local, dir_unit);
            s = norm(vec);
            if s < 1e-6
                R_align = eye(3);
            else
                c = dot(n_local, dir_unit);
                vx = [0 -vec(3) vec(2);
                    vec(3) 0 -vec(1);
                    -vec(2) vec(1) 0];
            
                R_align = eye(3) + vx + vx^2 * ((1 - c) / s^2);
            end
            
            K = [0 -dir_unit(3) dir_unit(2); 
                dir_unit(3) 0 -dir_unit(1); 
                -dir_unit(2) dir_unit(1) 0];
            
            R_theta = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
            
            % 
        end


        function disp(pyr)
            disp(['<a href="matlab:help Pyramid">Pyramid</a> [' int2str(pyr.size) '] : v a psi']);
            disp([reshape(pyr.v.X,1,pyr.numel());
                  reshape(pyr.v.Y,1,pyr.numel());
                  reshape(pyr.v.Z,1,pyr.numel());
                  reshape(pyr.v.Vx,1,pyr.numel());
                  reshape(pyr.v.Vy,1,pyr.numel());
                  reshape(pyr.v.Vz,1,pyr.numel());
                  reshape(pyr.a,1,pyr.numel());  
                  reshape(pyr.psi,1,pyr.numel())]);
        end

        function p = intersectionpoint(pyr, d, n)
            % INTERSECTIONPOINT Intersection point between pyramid and a line/ray/vector
            % P = INTERSECTIONPOINT(PYR,D,N) computes the N-th intersection point
            % between a pyramid and a line/vector/ray. If no intersection, returns NaN.
        
            % Input validation
            Check.isa('D must be a SLine, Vector or Ray', d, 'SLine', 'Vector', 'Ray');
            Check.isinteger('N must be 1 or 2', n, '>=', 1, '<=', 2);
        
            % Convert to line
            if isa(d, 'SLine')
                ln = d;
            else
                ln = d.toline();
            end
        
            % Pyramid center and rotation
            tr = Point(pyr.v.X, pyr.v.Y, pyr.v.Z);
            [R_align, R_theta] = RMatrix(pyr);
            R = R_theta * R_align;
        
            ln_tr = ln.translate(-tr);
            rotated_p1 = rotate_point(ln_tr.p1, R);
            rotated_p2 = rotate_point(ln_tr.p2, R);
            ln2 = SLine(rotated_p1, rotated_p2);
            dir = ln2.p2 - ln2.p1;
        
            % Pyramid parameters in local frame
            a_h = pyr.a / 2;
            h_cm = pyr.v.norm() / 3;
            apex = [0, 0, 3 * h_cm];
            base = [-a_h, -a_h, -h_cm;
                     a_h, -a_h, -h_cm;
                     a_h,  a_h, -h_cm;
                    -a_h,  a_h, -h_cm];
        
            [t_min, t_max] = intersections(ln2, base, apex, h_cm);
        
            if t_min > t_max
                p = Point(NaN, NaN, NaN);
                return;
            end
        
            t = t_min;
            if n == 2
                t = t_max;
            end
        
            p_local = ln2.p1 + t .* dir;
            p_rotated_back = rotate_point(p_local, R');
            p = p_rotated_back.translate(tr);

            function rotated_point = rotate_point(pt, R)
            sz = size(pt);
            num_points = numel(pt.X);
            coords = [reshape(pt.X, 1, num_points);
                      reshape(pt.Y, 1, num_points);
                      reshape(pt.Z, 1, num_points)];
            rotated_coords = R' * coords;
            rotated_point = Point(reshape(rotated_coords(1,:), sz), ...
                                  reshape(rotated_coords(2,:), sz), ...
                                  reshape(rotated_coords(3,:), sz));
            end
            
            function [t_min, t_max] = intersections(ln2, base, apex, h_cm)
                sz = size(ln2.p1.X);
                dir = ln2.p2 - ln2.p1;
                t_min = zeros(sz);
                t_max = ones(sz);
                eps = 1e-12;
            
                for i = 1:4
                    v0 = apex;
                    v1 = base(i,:);
                    v2 = base(mod(i,4)+1,:);
                    e1 = v1 - v0;
                    e2 = v2 - v0;
            
                    edge1 = repmat(reshape(e1, 1, 1, 3), [sz 1]);
                    edge2 = repmat(reshape(e2, 1, 1, 3), [sz 1]);
                    dir_mat = cat(3, dir.X, dir.Y, dir.Z);
                    h_mat = cross(dir_mat, edge2, 3);
                    A = dot(edge1, h_mat, 3);
            
                    f = 1.0 ./ A;
                    b = cat(3, ln2.p1.X - v0(1), ln2.p1.Y - v0(2), ln2.p1.Z - v0(3));
                    u = f .* dot(b, h_mat, 3);
                    q = cross(b, edge1, 3);
                    w = f .* dot(dir_mat, q, 3);
                    t = f .* dot(edge2, q, 3);
            
                    valid = abs(A) > eps & u >= 0 & w >= 0 & (u + w) <= 1 & t >= 0 & t <= 1;
            
                    t_min(valid) = max(t_min(valid), t(valid));
                    t_max(valid) = min(t_max(valid), t(valid));
                end
            
                z_parallel = abs(dir.Z) < eps;
                invalid_z = z_parallel & (ln2.p1.Z < -h_cm | ln2.p1.Z > -h_cm);
                t_min(invalid_z) = 1;
                t_max(invalid_z) = 0;
            
                valid_z = ~z_parallel;
                t = (-h_cm - ln2.p1.Z) ./ dir.Z;
                t_min(valid_z) = max(t_min(valid_z), t(valid_z));
                t_max(valid_z) = min(t_max(valid_z), t(valid_z));
            end
        end
        
        



        function ln = perpline(pyr,p)
            % PERPLINE Line perpendicular to pyramid faces (approximate)
            % Returns a placeholder line

            ln = SLine(p,p.translate(Vector(0,0,1))); % placeholder line
        end

        function pl = tangentplane(pl,~)
            % TANGENTPLANE Returns the plane itself (placeholder)
            % PL = TANGENTPLANE(PL,P)

            % For pyramid (discrete surfaces), we keep this dummy method
        end

        

    end
end
