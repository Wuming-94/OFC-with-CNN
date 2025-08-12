% Series of examples to demonstrate the use of ParticleSpherical.
%
% See also Particle, ParticleSpherical.
%
% The OTGO - Optical Tweezers in Geometrical Optics
% software package complements the article by
% Agnese Callegari, Mite Mijalkov, Burak Gokoz & Giovanni Volpe
% 'Computational toolbox for optical tweezers in geometrical optics'
% (2014).

%   Author: Giovanni Volpe
%   Version: 1.0.0
%   Date: 2014/01/01


% example('Use of ParticleSpherical')



%% DEFINITION OF PARTICLESPHERICAL
exampletitle('DEFINITION OF PARTICLEpyramid')
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

examplecode('v= Vector(5e-6,0,4e-6,0,0,6e-6);')
examplecode('a = 8e-6;')
examplecode('psi= 0;')
% examplecode('nm = 1;')
% examplecode('np = 1.5;')
examplecode('bead = ParticlePyramid(v,a,psi,nm,np)')
examplewait()

%% PLOTTING OF PARTICLESPHERICAL
exampletitle('PLOTTING OF PARTICLEpyramid')

figure
title('PARTICLEpyramid')
hold on
axis equal
grid on
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
view(30,30) % 更好的3D视角
axis([-15e-6 15e-6 -15e-6 15e-6 -15e-6 15e-6]) % 5微米范围

examplecode('bead.plot;') 
examplewait()

%% SCATTERING
exampletitle('SCATTERING')

exampletitle('SCATTERING')

examplecode('mr = 3;')
examplecode('nr = 2;')
examplecode('v = Vector(zeros(mr,nr),zeros(mr,nr),zeros(mr,nr),rand(mr,nr),rand(mr,nr),rand(mr,nr));')
examplecode('P = ones(mr,nr);')
examplecode('pol = Vector(zeros(mr,nr),zeros(mr,nr),zeros(mr,nr),ones(mr,nr),ones(mr,nr),ones(mr,nr)); pol = v*pol;')
examplecode('r = Ray(v,P,pol);')
examplewait()

examplecode('r.plot(''color'',''k'');')
examplewait()% examplecode('bg = BeamGauss(Ex0,Ey0,w0,L,Nphi,Nr);')
% examplecode('bg = bg.normalize(power);')
% examplecode('r = Ray.beam2focused(bg,f);')
% examplewait()
% 
% examplecode('r.plot(''color'',''y'');')
axis equal
grid on
view(3)
xlabel('x')
ylabel('y')
zlabel('z')
view(30,30) % 更好的3D视角
axis([-15e-6 15e-6 -15e-6 15e-6 -15e-6 15e-6]) % 5微米范围

examplewait()

examplecode('r_vec = bead.scattering(r)')
examplewait()

examplecode('rr = r_vec(1).r;')
examplecode('rr.plot(''color'',''r'');') %reflect
examplecode('rt = r_vec(1).t;')
examplecode('rt.plot(''color'',''b'');') %transmission
examplewait()

examplecode('rr = r_vec(2).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(2).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(3).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(3).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(4).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(4).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(5).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(5).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

%% FORCE
exampletitle('FORCE')

examplecode('F = bead.force(r)*1e+15 % fN')
examplewait()

%% TORQUE
exampletitle('TORQUE')

examplecode('T = bead.torque(r,1e-21)*1e+21 % fN*nm')