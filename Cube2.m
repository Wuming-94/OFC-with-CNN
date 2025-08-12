classdef Cube2 < Superficies
    % Cube < Superficies : A 3D cube defined by center, edge length, and rotation angles
    %
    % Properties:
    %   v      - center,orientation vectors (Vector)
    %   psi    - xrotation angles (roll)
    %
    % Methods:
    %   Cube                 - constructor
    %   plot                 - draw cube
    %   intersectionpoint    - ray intersection with cube
    %   perpline             - perpendicular line at point (normal)
    %   translate            - 3D translation
    %   xrotation            - rotation around x-axis
    %   yrotation            - rotation around y-axis
    %   zrotation            - rotation around z-axis
    %   numel                - number of cubes
    %   size                 - size of cube set
    %   disp                 - prints cube set

    properties
        v       % vector object
        psi     % xrotation angles (roll)
    end
    
    methods
        function obj = Cube(v, psi)
            % CUBE(v,psi) constructs a set of cubes with center v, edge length L, and rotation angles psi
            %   v must be a Vector
            %   psi must be a real scalar matrix with the same size as v
            
            Check.isa('v must be a Vector', v, 'Vector');
         
            Check.isreal('psi must be a real ', psi);
            Check.samesize('v and psi must have the same size', v, psi);
            
            obj.v = v;
            obj.psi = psi;
        end
        
        function h = plot(cb, varargin)
            % Scaling factor
            S = 1;
            for n = 1:2:length(varargin)
                if strcmpi(varargin{n}, 'scale')
                    S = varargin{n+1};
                    Check.isreal('The scaling factor must be a positive real number', S, '>', 0);
                end
            end
            
            % Get cube parameters
             Scale = 1e6;

            L = 2*cb.v.norm().* Scale;%lengt
            % center = [cb.v.X, cb.v.Y, cb.v.Z];

            [R_align,R_theta] = RMatrix(cb);
            R = R_align * R_theta;

            % Create cube vertices
            vertices = zeros(8, 3, numel(cb));

            for i = 1:numel(cb)
                % Cube vertices in local coordinates (centered at origin)
               a = L(i) / 2;
                v_local = [
                   -a, -a, -a;
                    a, -a, -a;
                    a, -a, a;
                    -a, -a, a;
                    -a, a, -a;
                    a, a, -a;
                    a, a, a;
                    -a, a, a
                ];
                

                % Rotate vertices
                v_rotated = (R * v_local')';
                
                % Translate to cube center
                center = [cb.v.X(i), cb.v.Y(i), cb.v.Z(i)];
                vertices(:,:,i) = v_rotated + center;

            end
            
            % Define cube faces
            faces = [
                1, 2, 3, 4; % Front
                5, 6, 7, 8; % Back
                1, 4, 8, 5; % Left
                2, 3, 7, 6; % Right
                1, 2, 6, 5; % Bottom
                3, 4, 8, 7  % Top
            ];

             colors = [
                1.0 0.2 0.2;  % 红
                0.6 0.6 0.6;  % 灰 
                0.2 0.2 1.0;  % 蓝
                0.2 0.2 1.0;  % 蓝
                1.0 1.0 0.2;  % 黄
                0.2 1.0 0.2;  % 绿
            ];
            
            % Plot cubes
            h = zeros(size(cb));
            for i = 1:numel(cb)
                     % Plot the cube
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
            
            % Set additional properties
            for n = 1:2:length(varargin)
                if ~strcmpi(varargin{n}, 'scale')
                    set(h, varargin{n}, varargin{n+1});
                end
            end
            
            % Output if needed
            if nargout > 0
                h = h;
            end
        end
    

        
        function disp(cb)
            % DISP Prints cube set
            %
            % DISP(CUBE) prints set of cubes CUBE.
            
            disp(['<a href="matlab:help Cube">Cube</a> [' int2str(cb.size) '] : X Y Z Vx Vy Vz Psi']);
            disp([reshape(cb.v.X, 1, cb.numel()); reshape(cb.v.Y, 1, cb.numel()); reshape(cb.v.Z, 1, cb.numel()); ...
                  reshape(cb.v.Vx, 1, cb.numel()); reshape(cb.v.Vy, 1, cb.numel()); reshape(cb.v.Vz, 1, cb.numel()); ...
                   reshape(cb.psi, 1, cb.numel())]);
        end

       function [R_align , R_theta] = RMatrix(cb)
         
            dir = [cb.v.Vx, cb.v.Vy, cb.v.Vz];
            theta = cb.psi;

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

        function cube_t = translate(cb, dp)
            % TRANSLATE 3D translation of cube set
            %
            % CUBE_T = TRANSLATE(CUBE,dP) translates set of cubes CUBE by dP.
            %   If dP is a Point, the translation corresponds to the
            %   coordinates X, Y and Z.
            %   If dP is a Vector, the translation corresponds to the
            %   components Vx, Vy and Vz.
            
            Check.isa('dP must be either a Point or a Vector', dp, 'Point', 'Vector');
            
            cube_t = cb;
            cube_t.v = cb.v.translate(dp);
        end
        
        function cube_r = xrotation(cb, phi)
            % XROTATION Rotation around x-axis of cube set
            %
            % CUBE_R = XROTATION(CUBE,phi) rotates set of cubes CUBE around x-axis 
            %   by an angle phi [rad].
            
            Check.isreal('The rotation angle phi must be a real number', phi);
            
            cube_r = cb;
            cube_r.v = cb.v.xrotation(phi);
           
        end
        
        function cube_r = yrotation(cb, phi)
            % YROTATION Rotation around y-axis of cube set
            %
            % CUBE_R = YROTATION(CUBE,phi) rotates set of cubes CUBE around y-axis 
            %   by an angle phi [rad].
            
            Check.isreal('The rotation angle phi must be a real number', phi);
            
            cube_r = cb;
            cube_r.v = cb.v.yrotation(phi);

        end
        
        function cube_r = zrotation(cb, phi)
            % ZROTATION Rotation around z-axis of cube set
            %
            % CUBE_R = ZROTATION(CUBE,phi) rotates set of cubes CUBE around z-axis 
            %   by an angle phi [rad].
            
            Check.isreal('The rotation angle phi must be a real number', phi);
            
            cube_r = cb;
            cube_r.v = cb.v.zrotation(phi);
            

        end
        
        function n = numel(cb)
            % NUMEL Number of cubes
            %
            % N = NUMEL(CUBE) number of cubes in set CUBE.
            
            n = numel(cb.v);
        end
        
        function s = size(cb, varargin)
            % SIZE Size of the cube set
            % 
            % S = SIZE(CUBE) returns a two-element row vector with the number 
            %   of rows and columns in the cube set CUBE.
            %
            % S = SIZE(CUBE,DIM) returns the length of the dimension specified 
            %   by the scalar DIM in the cube set CUBE.
            
            if ~isempty(varargin)
                s = cb.v.size(varargin{1});
            else
                s = cb.v.size();
            end
        end
        
        function p = intersectionpoint(cb,d,n)
            % INTERSECTIONPOINT Intersection point between cube and line/vector/ray
            %
            % P = INTERSECTIONPOINT(cb,D,N) calculates intersection points 
            %   between a set of lines (or vectors) D and the set of cube cb.
            %   The intersection point is selected by  N = {1,2}.
            %   If D does not intersect cb, the coordiantes of P are NaN.
            % 
            % See also Cube, Point, Vector, SLine, Ray.
    
            Check.isa('D must be a SLine, a Vector or a Ray',d,'SLine','Vector','Ray')
            Check.isinteger('N must be either 1 or 2',n,'>=',1,'<=',2)
        
            if isa(d, 'SLine')
                ln = d;
            else
                ln = d.toline();                
            end
        
            % 1. 转换到立方体的局部坐标系（关键修改部分）
            tr = Point(cb.v.X, cb.v.Y, cb.v.Z);  % 立方体中心
            [R_align , R_theta] = cb.RMatrix();  % 获取立方体的旋转矩阵（Cube类的方法）
            R = R_align * R_theta;
            a = cb.v.norm();

            % 2. 先将线段平移到原点（减去立方体中心）
            ln_tr = ln.translate(-tr);


            rotated_p1 = rotate_point(ln_tr.p1, R);
            rotated_p2 = rotate_point(ln_tr.p2, R);
    
            % 3. 创建旋转后的线段集合
            ln2 = SLine(rotated_p1, rotated_p2);% 局部坐标系下的线段
            % ln2.plot;
            
        
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

          %% 2. slab Method
            [t_min, t_max] = compute_intersection_parameters(ln2, a);
            
            % 判断是否有交点
            if t_min > t_max
                % 无交点，返回NaN
                p = Point(NaN, NaN, NaN);
                return;
            end
            
            % 根据n选择对应的t值
            if n == 1
                t = t_min;
            else
                t = t_max;
            end
            
            % 计算局部坐标系下的交点
            dir = ln2.p2 - ln2.p1;  % 线段方向向量
            p_local = ln2.p1 + t* dir;
            
            
            % 3. 转换回原坐标系
            % 先逆旋转（旋转矩阵的转置即为逆矩阵）
            p_rotated_back = rotate_point(p_local, R');
            % 再逆平移
            p = p_rotated_back.translate(tr);

        
          %% 射线参数：起点p1，方向向量dir（局部坐标系下）
          function [t_min, t_max] = compute_intersection_parameters(ln2, a)
            % 计算线段与轴对齐正方体的交点参数t
            % ln: 局部坐标系下的线段
            % a: 正方体半边长
            % 返回t的有效范围[t_min, t_max]
            
            p1 = ln2.p1;
            p2 = ln2.p2;
            dir = p2 - p1;  % 线段方向向量
            
            % t: Range ln in Cube [t_min,t_max]
            t_min = 0;
            t_max = 1;
            
            % 处理x方向平面 (x = -a 和 x = a)
            if abs(dir.X) > eps  % MATLAB 中的一个极小值（约 2.2e-16）
                t1 = (-a - p1.X) ./ dir.X;  % 与x=-a平面的交点参数
                t2 = (a - p1.X) ./ dir.X;   % 与x=a平面的交点参数
                t_x_min = min(t1, t2);
                t_x_max = max(t1, t2); % Range [t_x_min,t_x_max] ln in Cube 
                
                % 更新全局t范围
                t_min = max(t_min, t_x_min);  % corange, minimal range ensure all dim in Cube.
                t_max = min(t_max, t_x_max);
                %when dim y paralle to Cube surface, t_min = 1; t_max = 0; 
                % which result in t_min > t_max,no intersectionpoints.
            else
                % in yz plane
                if p1.X < -a | p1.X > a
                    % 线段不在x范围内，无交点
                    t_min = 1;
                    t_max = 0;
                    return;
                end
            end
            
            % 处理y方向平面 (y = -a 和 y = a)
            if abs(dir.Y) > eps  % Unparalle
                t1 = (-a - p1.Y) ./ dir.Y;  % 与y=-a平面的交点参数
                t2 = (a - p1.Y) ./ dir.Y;   % 与y=a平面的交点参数
                t_y_min = min(t1, t2);
                t_y_max = max(t1, t2);
                
                % 更新全局t范围
                t_min = max(t_min, t_y_min);
                t_max = min(t_max, t_y_max);
            else
                % in xz plane
                if p1.Y < -a | p1.Y > a
                    % 线段不在y范围内，无交点
                    t_min = 1;
                    t_max = 0;
                    return;
                end
            end
            
            % 处理z方向平面 (z = -a 和 z = a)
            if abs(dir.Z) > eps
                t1 = (-a - p1.Z) ./ dir.Z;  % 与z=-a平面的交点参数
                t2 = (a - p1.Z) ./ dir.Z;   % 与z=a平面的交点参数
                t_z_min = min(t1, t2);
                t_z_max = max(t1, t2);
                
                % 更新全局t范围
                t_min = max(t_min, t_z_min);
                t_max = min(t_max, t_z_max);
            else
                % in x-y plane,z no difference
                if p1.Z < -a | p1.Z > a
                    % 线段不在z范围内，无交点
                    t_min = 1;
                    t_max = 0;
                    return;
                end
            end
          end
       end

       function ln = perpline(cb, p)
            % PERPLINE Line perpendicular to cube surfaces passing through point p
            %
            % LN = PERPLINE(CB,P) calculates lines LN that are perpendicular
            % to cube CB and pass through points P. The function considers
            % face, edge, and corner proximity for direction assignment.
        
            Check.isa('P must be a Point', p, 'Point')

            [R_align , R_theta] = cb.RMatrix();  % 3x3旋转矩阵
            R = R_align * R_theta;
            a = cb.v.norm();   % 半边长
            
            % 立方体中心坐标，作为3x1向量
            c_vec = [cb.v.X; cb.v.Y; cb.v.Z];  

            c = Point(cb.v.X.*ones(size(p)),cb.v.Y.*ones(size(p)),cb.v.Z.*ones(size(p))); % cube center
            cp = SLine(c,p).tovector();
            
            % 局部主轴单位向量
            e1 = [1,0,0];
            e2 = [0,1,0];
            e3 = [0,0,1];
            
            % 旋转后主轴向量（单位向量）
            cb1 = R * e1'+c_vec; 
            cb2 = R * e2'+c_vec; 
            cb3 = R_align * e3'+c_vec;
            
            % % 立方体顶点示例（中心加上半边长方向的向量）
            v1 = Vector(cb.v.X, cb.v.Y, cb.v.Z, a * cb1(1),a * cb1(2),a * cb1(3));
            v2 = Vector(cb.v.X, cb.v.Y, cb.v.Z, a * cb2(1),a * cb2(2),a * cb2(3));
            v3 = Vector(cb.v.X, cb.v.Y, cb.v.Z, a * cb3(1),a * cb3(2),a * cb3(3));
            
            % 归一化（可选，cbv1~cbv3已经是单位向量）
            cbv1 = v1.versor();
            cbv2 = v2.versor();
            cbv3 = v3.versor();

            
            cpx = (cp.*cbv1)*cbv1; % component parallel to local x axis
            cpy = (cp.*cbv2)*cbv2; % component parallel to local y axis
            cpz = (cp.*cbv3)*cbv3; % component parallel to local z axis

            perpendicularx = cp-cpx;  % component pendicular to cpx
            perpendiculary = cp-cpy;  % component pendicular to cpx
            perpendicularz = cp-cpz;  % component pendicular to cpx

            ln_x = SLine(perpendicularx.toline().p2,p); %pendicular to face x=a/-a with p
            ln_y = SLine(perpendiculary.toline().p2,p); %pendicular to face y=a/-a with p
            ln_z = SLine(perpendicularz.toline().p2,p); %pendicular to face z=a/-a with p

            ln = SLine(c,p);

            % p in x= ±a 
            pinxf =  cpx.norm()>=a & cpy.norm()<a & cpz.norm()<a;

            ln.p1.X(pinxf) = ln_x.p1.X(pinxf); %p2
            ln.p1.Y(pinxf) = ln_x.p1.Y(pinxf);
            ln.p1.Z(pinxf) = ln_x.p1.Z(pinxf);
            ln.p2.X(pinxf) = ln_x.p2.X(pinxf); %p
            ln.p2.Y(pinxf) = ln_x.p2.Y(pinxf);
            ln.p2.Z(pinxf) = ln_x.p2.Z(pinxf);

            % p in y= ±a 
            pinyf =  cpy.norm()>=a & cpx.norm()<a & cpz.norm()<a;

            ln.p1.X(pinyf) = ln_y.p1.X(pinyf); %p2
            ln.p1.Y(pinyf) = ln_y.p1.Y(pinyf);
            ln.p1.Z(pinyf) = ln_y.p1.Z(pinyf);
            ln.p2.X(pinyf) = ln_y.p2.X(pinyf); %p
            ln.p2.Y(pinyf) = ln_y.p2.Y(pinyf);
            ln.p2.Z(pinyf) = ln_y.p2.Z(pinyf);

            % p in z= ±a 
            pinzf =  cpz.norm()>=a & cpy.norm()<a & cpx.norm()<a;

            ln.p1.X(pinzf) = ln_z.p1.X(pinzf); %p2
            ln.p1.Y(pinzf) = ln_z.p1.Y(pinzf);
            ln.p1.Z(pinzf) = ln_z.p1.Z(pinzf);
            ln.p2.X(pinzf) = ln_z.p2.X(pinzf); %p
            ln.p2.Y(pinzf) = ln_z.p2.Y(pinzf);
            ln.p2.Z(pinzf) = ln_z.p2.Z(pinzf);

            % p in edge 
            pinedge = (cpx.norm()>=a & cpy.norm()>=a) | ...
                      (cpx.norm()>=a & cpz.norm()>=a) | ...
                      (cpy.norm()>=a & cpz.norm()>=a);

            ln.p1.X(pinedge) = NaN;
            ln.p1.Y(pinedge) = NaN;
            ln.p1.Z(pinedge) = NaN;
            ln.p2.X(pinedge) = NaN;
            ln.p2.Y(pinedge) = NaN;
            ln.p2.Z(pinedge) = NaN;

        end
  
        function pl = tangentplane(cb,p)
            % TANGENTPLANE Plane tangent to cube passing by point
            % 
            % PL = TANGENTPLANE(cb,P) calculates plane set PLt tangent to 
            %   cube cb and passing by points P.
            % 
            % See also cube, Point, Plane.

            Check.isa('P must be a Point',p,'Point')

            pl = Plane.perpto(cb.perpline(p),p);
        end
        
    end
        
end