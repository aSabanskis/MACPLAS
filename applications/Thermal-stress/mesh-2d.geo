// A simple rectangular block

Lx = 0.01;
Ly = 0.10;

Nx = 11;
Ny = 101;

Point(1) = {0,  -Ly/2, 0};
Point(2) = {Lx, -Ly/2, 0};
Point(3) = {Lx,  Ly/2, 0};
Point(4) = {0,   Ly/2, 0};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(5) = {1:4};
Plane Surface(6) = {5};

Transfinite Line {1,3} = Nx+1;
Transfinite Line {2,4} = Ny+1;

Transfinite Surface "*";
Recombine Surface "*";
