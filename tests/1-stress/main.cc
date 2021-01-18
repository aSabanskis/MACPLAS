#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include "../../include/stress_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order = 1);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  double r_a, r_b;
  double T_a, T_b;

  const SphericalManifold<dim> manifold;

  Vector<double> temperature;

  StressSolver<dim> solver;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order)
  : r_a(0.5)
  , r_b(1.0)
  , T_a(500)
  , T_b(1000)
  , manifold(SphericalManifold<dim>(Point<dim>()))
  , solver(order)
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  solver.solve();
  solver.output_vtk();

  // write results on x axis
  std::stringstream ss;
  ss << "result-" << dim << "d-order"
     << solver.get_dof_handler().get_fe().degree << "-x.dat";
  const std::string file_name = ss.str();
  std::cout << "Saving postprocessed results to '" << file_name << "'";

  const Vector<double> &     temperature = solver.get_temperature();
  const BlockVector<double> &stress      = solver.get_stress();

  std::ofstream f(file_name);
  f << (dim == 2 ? "x y" : "x y z");
  f << " T[K]";
  for (unsigned int k = 0; k < stress.n_blocks(); ++k)
    f << " stress_" << k << "[Pa]";
  f << '\n';
  f << std::setprecision(8);

  std::vector<Point<dim>> support_points;
  solver.get_support_points(support_points);

  for (unsigned int i = 0; i < support_points.size(); ++i)
    {
      // select points with y=z=0
      Point<dim> p = support_points[i];
      p[0]         = 0;
      if (p.norm_square() > 1e-8)
        continue;

      f << support_points[i] << " " << temperature[i];
      for (unsigned int k = 0; k < stress.n_blocks(); ++k)
        f << " " << stress.block(k)[i];
      f << '\n';
    }
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Assert(r_a > 0, ExcMessage("r_a must be positive"));
  Assert(r_b > r_a, ExcMessage("r_b must be larger than r_b"));

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::quarter_hyper_shell(
    triangulation, Point<dim>(), r_a, r_b, 0, true);

  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold(0, manifold);
  triangulation.set_manifold(1, manifold);

  triangulation.refine_global(3);
}

template <int dim>
void
Problem<dim>::initialize()
{
  // Set the temperature using analytical solution in spherical coordinates
  // T(r) = c1 / r + c2
  const double c1 = r_a * (T_a - T_b) / (1 - r_a / r_b);
  const double c2 = (T_b - T_a * r_a / r_b) / (1 - r_a / r_b);

  std::cout << "r_a=" << r_a << "\n"
            << "r_b=" << r_b << "\n"
            << "T_a=" << T_a << "\n"
            << "T_b=" << T_b << "\n"
            << "c1=" << c1 << "\n"
            << "c2=" << c2 << "\n";

  solver.initialize();

  std::vector<Point<dim>> support_points;
  solver.get_support_points(support_points);

  Vector<double> &temperature = solver.get_temperature();
  AssertDimension(temperature.size(), support_points.size());

  for (unsigned int i = 0; i < temperature.size(); ++i)
    {
      temperature[i] = c1 / support_points[i].norm() + c2;
    }

  // set symmetry boundary conditions
  // u_x(x=0) = u_y(y=0) = u_z(z=0) = 0
  for (unsigned int k = 0; k < dim; ++k)
    {
      // for boundary numbering, see documentation of quarter_hyper_shell
      solver.set_bc1(k + 2, k, 0);
    }
}

int
main()
{
  deallog.attach(std::cout);
  deallog.depth_console(2);

  for (unsigned int order = 1; order <= 2; ++order)
    {
      Problem<2> p2d(order);
      p2d.run();

      Problem<3> p3d(order);
      p3d.run();
    }

  return 0;
}
