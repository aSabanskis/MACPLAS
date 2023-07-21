R = 0.01;
nR = 16;

z0 = 0;
z1 = 0.03;
z2 = 0.474;
z3 = 0.4885;

dz = 5e-4;
nZ1 = Ceil(Hypot(z1-z0, R)/dz);
nZ2 = Ceil((z2-z1)/dz);
nZ3 = Ceil(Hypot(z3-z2, R)/dz);

Printf("nZ1=%g, nZ2=%g, nZ3=%g", nZ1, nZ2, nZ3);

Point(1) = {0,    z0, 0};
Point(2) = {1e-3, z0, 0};
Point(3) = {R,    z1, 0};
Point(4) = {R,    z2, 0};
Point(5) = {2e-3, z3, 0};
Point(6) = {0,    z3, 0};
Point(7) = {0,    z2, 0};
Point(8) = {0,    z1, 0};

nSpline1 = newp;
Point(nSpline1+0) = {4.512e-3,  6.767e-3, 0};
Point(nSpline1+1) = {7.342e-3, 12.565e-3, 0};
Point(nSpline1+2) = {8.735e-3, 16.766e-3, 0};
Point(nSpline1+3) = {9.432e-3, 20.779e-3, 0};
Point(nSpline1+4) = {9.834e-3, 24.532e-3, 0};
Point(nSpline1+5) = {9.978e-3, 27.478e-3, 0};

nSpline2 = newp;
Point(nSpline2+0) = {9.9e-3, 475e-3, 0};
Point(nSpline2+1) = {9.2e-3, 477.037e-3, 0};
Point(nSpline2+2) = {7.311e-3, 479.654e-3, 0};
Point(nSpline2+3) = {5.466e-3, 482.295e-3, 0};
Point(nSpline2+4) = {3.957e-3, 484.865e-3, 0};
Point(nSpline2+5) = {2.735e-3, 486.955e-3, 0};

Line(1) = {1, 2};
Spline(2) = {2, nSpline1:nSpline1+5, 3};
Line(3) = {3, 4};
Spline(4) = {4, nSpline2:nSpline2+5, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {8, 3};
Line(10) = {7, 4};

Line Loop(1) = {1, 2, -9, 8};
Line Loop(2) = {3, -10, 7, 9};
Line Loop(3) = {4, 5, 6, 10};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Line(0) = {1,2,3,4,5};
Physical Line(1) = {6,7,8};
Physical Surface(0) = {1,2,3};

Transfinite Line {1,5} = nR;
Transfinite Line {9,10} = nR Using Progression 0.9;
Transfinite Line {2,8} = nZ1;
Transfinite Line {4,6} = nZ3;
Transfinite Line {3,7} = nZ2;
Transfinite Surface "*";

Recombine Surface "*";
