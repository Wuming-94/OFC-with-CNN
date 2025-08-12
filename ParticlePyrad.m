classdef ParticlePyramid < Particle
    % ParticlePyramid < Particle : Optically trappable square pyramid
    %
    % Properties:
    %   center  - Point object (center of the base)
    %   a       - Base edge length
    %   h       - Height (from base center to tip)
    %   dir     - Direction vector from base center to tip
    %   theta   - Rotation around direction vector
    %
    % Methods:
    %   plot        - 3D plot of the pyramid
    %   barycenter  - Return base center point
    %   translate   - Move pyramid in 3D
    %   xrotation, yrotation, zrotation - Axis rotations

    properties
        center  % Point object (base center)
        a       % Base edge length
        h       % Height
        dir     % 3x1 vector (direction)
        theta   % Rotation angle around dir
    end

    methods
        function obj = ParticlePyramid(center, a, h, dir, theta)
            % Constructor
            %
            % ParticlePyramid(C, a, h, dir, theta)
            %   center : Point object (base center)
            %   a      : base edge length
            %   h      : height
            %   dir    : direction vector (3x1)
            %   theta  : rotation angle [rad]

            Check.isa('Center must be a Point', center, 'Point');
            Check.isreal('a must be a positive real', a, '>', 0);
            Check.isreal('h must be a positive real', h, '>', 0);
            Check.isnumeric('dir must be numeric 3x1 vector', dir);
            Check.isreal('theta must be real', theta);

            obj.center = center;
            obj.a = a;
            obj.h = h;
            obj.dir = dir;
            obj.theta = theta;
        end

        function h_out = plot(obj)
            % PLOT Visualize the pyramid in 3D
            %
            % h_out = PLOT(obj)

            h_out = plotPyramidWithPose(obj.center.toarray(), obj.a, obj.h, obj.dir, obj.theta);
        end

        function p = barycenter(obj)
            % BARYCENTER Return base center point
            %
            % p = BARYCENTER(obj)

            p = obj.center;
        end

        function obj2 = translate(obj, dp)
            % TRANSLATE Move pyramid
            %
            % obj2 = TRANSLATE(obj, dp)
            % dp : Vector or Point

            Check.isa('dp must be Point or Vector', dp, 'Point', 'Vector');
            obj2 = obj;
            obj2.center = obj.center + dp;
        end

        function obj2 = xrotation(obj, phi)
            % XROTATION Rotate around x-axis
            R = rotx(phi);
            obj2 = obj.rotateWithMatrix(R);
        end

        function obj2 = yrotation(obj, phi)
            % YROTATION Rotate around y-axis
            R = roty(phi);
            obj2 = obj.rotateWithMatrix(R);
        end

        function obj2 = zrotation(obj, phi)
            % ZROTATION Rotate around z-axis
            R = rotz(phi);
            obj2 = obj.rotateWithMatrix(R);
        end

        function obj2 = rotateWithMatrix(obj, R)
            % Internal helper to rotate direction + center
            obj2 = obj;
            obj2.dir = R * obj.dir;
            c = obj.center.toarray();
            obj2.center = Point(R * c);
        end
    end
end
