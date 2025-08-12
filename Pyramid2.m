classdef Pyramid2 < Superficies
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
        function obj = Pyramid2(v,a,psi)
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

            Scale = 1;       
         
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

            [R_align,R_theta] = RMatrix(pyr);
            
            R = R_align *R_theta;

            base_world = R * base_local + center';
            top_world  = R * top_local + center';

            vertices = [base_world, top_world];  % 4底+1顶，总5个点
            vertices = vertices';

            faces = [
                1 2 3 4;  % base
                1 2 5 NaN;  % front
                2 3 5 NaN;  % 
                3 4 5 NaN;  % 
                4 1 5 NaN;  % 
            ];
            
            colors = [
                0.6 0.6 0.6;  % base gray
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

        function [R_align , R_theta] = RMatrix(pyr)
         
            dir = [pyr.v.Vx, pyr.v.Vy, pyr.v.Vz];
            theta = pyr.psi;

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
            %
            % P = INTERSECTIONPOINT(PYR,D,N) computes the N-th intersection point
            %   between a pyramid and a line/vector/ray. If no intersection, returns NaN.
            
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

            [R_align,R_theta] = RMatrix(pyr);
            
            R =  R_align*R_theta;
        
            ln_tr = ln.translate(-tr);
           

            rotated_p1 = rotate_point(ln_tr.p1, R);
            rotated_p2 = rotate_point(ln_tr.p2, R);
    
            ln2 = SLine(rotated_p1, rotated_p2);% local rays
            dir = ln2.p2 - ln2.p1; 
            % ln2.plot('Color', [1 0.7 0.7]);    % 浅粉红
            % pyr.plot;

            % Pyramid parameters local
            a_h = pyr.a / 2;
            h_cm = pyr.v.norm() /3;

            %top point
            apex = [0, 0, 3 * h_cm]; %T
        
            % Base plane
            base = [-a_h, -a_h, -h_cm; %A
                     a_h, -a_h, -h_cm; %B
                     a_h,  a_h, -h_cm; %C
                    -a_h,  a_h, -h_cm];%D

            n_base = [0; 0; -1];
            
                function rotated_point = rotate_point(ln_tr, R)
                    sz = size(ln_tr);
                    rows = sz(1);
                    cols = sz(2);
    
                    num_points = rows * cols;
                    
                    % 转换为3xN矩阵以便批量处理
                    coords = [reshape(ln_tr.X, 1, num_points);
                              reshape(ln_tr.Y, 1, num_points);
                              reshape(ln_tr.Z, 1, num_points)];
                    
                    % 应用旋转矩阵
                    rotated_coords = R' * coords;
                    
                    % 重塑为原始尺寸并创建新的Point对象
                    rotated_point = Point(reshape(rotated_coords(1,:), rows, cols), ...
                                          reshape(rotated_coords(2,:), rows, cols), ...
                                          reshape(rotated_coords(3,:), rows, cols));
                end
    

            [hits, t_min,t_max] = intersections(ln2,base,apex,h_cm);

            no_hit = t_min > t_max;
            % 无交点，返回NaN
            p.X(no_hit) = NaN;
            p.Y(no_hit) = NaN;
            p.Z(no_hit) = NaN;

            function [hits, t_min, t_max] = intersections(ln2, base, apex, h_cm)
            % 初始化矩阵形式的参数
            [rows, cols] = size(ln2.p1.X);  % 80x40
            t_min = NaN(rows, cols);      % 初始进入参数
            t_max = NaN(rows, cols);       % 初始离开参数
            hits = false(rows, cols, 5);    % 5个面的击中情况 (3D 矩阵)
            
            % 光线方向 (矩阵形式)
            dir = ln2.p2 - ln2.p1;
            
            % ================== 处理4个侧面 ==================
            for i = 1:4
                v0 = apex;  % 金字塔顶点
                v1 = base(i, :);  % 底面第i点
                v2 = base(mod(i, 4)+1, :);  % 底面第i+1点
                
                % 向量化 Moller-Trumbore 算法
                e1 = v1 - v0;
                e2 = v2 - v0;
                
                % 扩展为 80x40x3 矩阵
                e1_mat = repmat(reshape(e1, [1, 1, 3]), rows, cols, 1);
                e2_mat = repmat(reshape(e2, [1, 1, 3]), rows, cols, 1);
                dir_mat = cat(3, dir.X, dir.Y, dir.Z);
                
                % 计算 h = cross(dir, e2)
                h_mat = cross(dir_mat, e2_mat, 3);
                
                % 计算 a = dot(e1, h)
                A = dot(e1_mat, h_mat, 3);
                
                % % 定义极小值
                % eps = 1e-12;
                
                % 计算 f = 1/a (避免除零)
                f = NaN(rows, cols);
                valid_a = abs(A) > eps;
                f(valid_a) = 1.0 ./ A(valid_a);
                
                % 计算从 v0 到光线起点的向量
                s = cat(3, ln2.p1.X - v0(1), ln2.p1.Y - v0(2), ln2.p1.Z - v0(3));
                u = f .* dot(s, h_mat, 3);
                q = cross(s, e1_mat, 3);
                w = f .* dot(dir_mat, q, 3);
                t = f .* dot(e2_mat, q, 3);
                vu= u >= 0 & u <= 1;
                vw = w >= 0 & (u + w) <= 1;
                vt = (t >= 0) & (t <= 1);

                valid = valid_a & vu & vw & vt;
                
                % 更新击中信息
                hits(:, :, i) = valid;
                
                % 仅更新有效光线的参数
                t_min(valid) = min(t_min(valid), t(valid),'omitnan');
                t_max(valid) = max(t_max(valid), t(valid),'omitnan');
            end
            
            valid_bottom = abs(dir.Z) > eps;
            t_base = NaN(rows, cols);
            
            % 计算交点参数
            t_base(valid_bottom) = (-h_cm - ln2.p1.Z(valid_bottom)) ./ dir.Z(valid_bottom);
            
            % 计算交点坐标
            x_base = ln2.p1.X + t_base .* dir.X;
            y_base = ln2.p1.Y + t_base .* dir.Y;
            
            % 底面边界 (假设底面是矩形)
            min_x = min(base(:, 1));
            max_x = max(base(:, 1));
            min_y = min(base(:, 2));
            max_y = max(base(:, 2));
            
            % 检查交点是否在底面内
            in_bottom = (x_base >= min_x) & (x_base <= max_x) & ...
                        (y_base >= min_y) & (y_base <= max_y) & ...
                        (t_base >= 0) & (t_base <= 1);
            
            % 更新底面击中信息
            hits(:, :, 5) = in_bottom;
            
            % 更新参数 (仅有效交点)
            t_min(in_bottom) = min(t_min(in_bottom), t_base(in_bottom),'omitnan');
            t_max(in_bottom) = max(t_max(in_bottom), t_base(in_bottom),'omitnan');
            
           end

            if n == 1
                t = t_min;
            else
                t = t_max;
            end

            p_local = ln2.p1 + t* dir;
            p_rotated_back = rotate_point(p_local, R');
            p = p_rotated_back.translate(tr);
    
        end


        function ln = perpline(pyr,p)
            % PERPLINE Line perpendicular to pyramid faces (approximate)
            % Returns a placeholder line
            Check.isa('P must be a Point', p, 'Point')

            [R_align , R_theta] = pyr.RMatrix();  % 3x3旋转矩阵
            R = R_align * R_theta;
            tr = Point(pyr.v.X, pyr.v.Y,pyr.v.Z);
            p_r = p.translate(-tr);
            [rows, cols] = size(p.X);  % 80x40

            function rotated_point = rotate_point(ln_tr, R)
                    sz = size(ln_tr);
                    rows = sz(1);
                    cols = sz(2);
    
                    num_points = rows * cols;
                    
                    % 转换为3xN矩阵以便批量处理
                    coords = [reshape(ln_tr.X, 1, num_points);
                              reshape(ln_tr.Y, 1, num_points);
                              reshape(ln_tr.Z, 1, num_points)];
                    
                    % 应用旋转矩阵
                    rotated_coords = R' * coords;
                    
                    % 重塑为原始尺寸并创建新的Point对象
                    rotated_point = Point(reshape(rotated_coords(1,:), rows, cols), ...
                                          reshape(rotated_coords(2,:), rows, cols), ...
                                          reshape(rotated_coords(3,:), rows, cols));
            end

            p_l = rotate_point(p_r, R');%p local

            A = p_l.Y< p_l.X & p_l.Y< -p_l.X & p_l.Y<0; % p in face A
            B = p_l.Y< p_l.X & p_l.Y> -p_l.X & p_l.X>0;
            C = p_l.Y> p_l.X & p_l.Y> -p_l.X & p_l.Y>0;
            D = p_l.Y> p_l.X & p_l.Y< -p_l.X & p_l.X<0;
            
            a_h = pyr.a / 2;
            h_cm = pyr.v.norm() /3;
            S = p_l.Z + h_cm < eps; %p in base


            %top point
            apex = [0, 0, 3 * h_cm]; %T
        
            % Base plane
            base = [-a_h, -a_h, -h_cm; %A
                     a_h, -a_h, -h_cm; %B
                     a_h,  a_h, -h_cm; %C
                    -a_h,  a_h, -h_cm];%D

            e1 = base(1,:) - apex;
            e2 = base(2,:) - apex;
            e3 = base(3,:) - apex;
            e4 = base(4,:) - apex;

            n_base = [0, 0, -1];

            nA = cross(e1,e2)/norm(cross(e1,e2));
            nB = cross(e2,e3)/norm(cross(e2,e3));
            nC = cross(e3,e4)/norm(cross(e3,e4));
            nD = cross(e4,e1)/norm(cross(e4,e1));

            % nA = cross(e1,e2);
            % nB = cross(e2,e3);
            % nC = cross(e3,e4);
            % nD = cross(e4,e1);
            % n_base = [0; 0; -h_cm];

            % 初始化 p2l 为 p_l 的拷贝
            p2l = p_l;
            
            % Face A
            p2l.X(A) = p_l.X(A) + nA(1);
            p2l.Y(A) = p_l.Y(A) + nA(2);
            p2l.Z(A) = p_l.Z(A) + nA(3);
            
            % Face B
            p2l.X(B) = p_l.X(B) + nB(1);
            p2l.Y(B) = p_l.Y(B) + nB(2);
            p2l.Z(B) = p_l.Z(B) + nB(3);
            
            % Face C
            p2l.X(C) = p_l.X(C) + nC(1);
            p2l.Y(C) = p_l.Y(C) + nC(2);
            p2l.Z(C) = p_l.Z(C) + nC(3);
            
            % Face D
            p2l.X(D) = p_l.X(D) + nD(1);
            p2l.Y(D) = p_l.Y(D) + nD(2);
            p2l.Z(D) = p_l.Z(D) + nD(3);
            
            % Base
            p2l.X(S) = p_l.X(S) + n_base(1);
            p2l.Y(S) = p_l.Y(S) + n_base(2);
            p2l.Z(S) = p_l.Z(S) + n_base(3);

            p2r = rotate_point(p2l, R);
            p2 = p2r.translate(tr);

            ln = SLine(p,p2);
        end

        function pl = tangentplane(pyr,p)
            % TANGENTPLANE Plane tangent to cube passing by point
            % 
            % PL = TANGENTPLANE(cb,P) calculates plane set PLt tangent to 
            %   cube cb and passing by points P.
            % 
            % See also cube, Point, Plane.

            Check.isa('P must be a Point',p,'Point')

            pl = Plane.perpto(pyr.perpline(p),p);
        end

    end
end
