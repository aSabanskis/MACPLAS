R1 = 0.002;

z0 = 0;
z1 = 0.008;
z2 = 0.008;

nR = 25;
nZ = 100;

qR = 1.08;
qZ = 1.02;

Point(1) = {0,  z0, 0};
Point(2) = {R1, z0, 0};
Point(3) = {R1, z1, 0};
Point(4) = {0,  z2, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};

Physical Line(0) = {1};
Physical Line(1) = {2,3};
Physical Line(2) = {4};
Physical Surface(0) = {1};

Transfinite Line {-1,3} = nR Using Progression qR;
Transfinite Line {2,-4} = nZ Using Progression qZ;
Transfinite Surface "*";

Recombine Surface "*";
