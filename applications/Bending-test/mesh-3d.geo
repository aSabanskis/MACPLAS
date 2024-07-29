// A simple block

Lx = 0.1;
Ly = 0.015;
Lz = 0.003;

Nx = 100;
Ny = 4;
Nz = 3;

z0 = -Lz/2;

Point(1) = {-Lx/2, -Ly/2, z0};
Point(2) = { Lx/2, -Ly/2, z0};
Point(3) = { Lx/2,  Ly/2, z0};
Point(4) = {-Lx/2,  Ly/2, z0};
Line(1) = {4,3};
Line(2) = {3,2};
Line(3) = {2,1};
Line(4) = {1,4};
Line Loop(5) = {2,3,4,1};
Plane Surface(6) = {5};

Transfinite Line {1,3} = Nx+1;
Transfinite Line {2,4} = Ny+1;

Transfinite Surface "*";
Recombine Surface "*";

tmp[] = Extrude {0, 0, Lz} {
  Surface{6};
  Layers{Nz};
  Recombine;
};

Physical Volume(1) = tmp[1];
