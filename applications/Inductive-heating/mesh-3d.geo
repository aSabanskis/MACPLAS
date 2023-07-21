scale45 = Sin(Pi/4);
R_outer = 0.01;
R_inner = 0.7*R_outer;
dR_inner = -0.8*R_inner;
L = 0.535;

n1 = 11;
n2 = 6;

Point(1) = {0, 0, 0};

Point(2) = { R_outer*scale45,  R_outer*scale45, 0};
Point(3) = {-R_outer*scale45,  R_outer*scale45, 0};
Point(4) = {-R_outer*scale45, -R_outer*scale45, 0};
Point(5) = { R_outer*scale45, -R_outer*scale45, 0};

Point(6) = { R_inner*scale45,  R_inner*scale45, 0};
Point(7) = {-R_inner*scale45,  R_inner*scale45, 0};
Point(8) = {-R_inner*scale45, -R_inner*scale45, 0};
Point(9) = { R_inner*scale45, -R_inner*scale45, 0};

Point(10) = {0,         dR_inner, 0};
Point(11) = {-dR_inner, 0,        0};
Point(12) = {0,        -dR_inner, 0};
Point(13) = { dR_inner, 0,        0};


Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Circle(5) = {6, 10, 7};
Circle(6) = {7, 11, 8};
Circle(7) = {8, 12, 9};
Circle(8) = {9, 13, 6};

Line(9)  = {2, 6};
Line(10) = {3, 7};
Line(11) = {4, 8};
Line(12) = {5, 9};

Line Loop(1) = {5, 6, 7, 8};
Line Loop(2) = {1, 10, -5, -9};
Line Loop(3) = {2, 11, -6, -10};
Line Loop(4) = {3, 12, -7, -11};
Line Loop(5) = {4,  9, -8, -12};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};

Transfinite Line {1:8} = n1;
Transfinite Line {9:12} = n2 Using Progression 1.3;

num[] = Extrude {0,0,L} {Surface{1:5}; Layers{ {14,5,6,5,5,10,10,6,6,10,10,5,5,6,5,14}, {0.28803738317757,0.34411214953271,0.40018691588785,0.437570093457944,0.465607476635514,0.484299065420561,0.495514018691589,0.5,0.504485981308411,0.515700934579439,0.534392523364486,0.562429906542056,0.59981308411215,0.65588785046729,0.71196261682243,1} }; Recombine;};

//Coherence;

Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";

Physical Surface(0) = {1:5,34,78,56,122,100,43,65,87,109};
Physical Volume(0) = {1:5};
