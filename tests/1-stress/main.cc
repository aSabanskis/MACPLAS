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
  solver.output_results();
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
  for (unsigned int order = 1; order <= 2; ++order)
    {
      Problem<2> p2d(order);
      p2d.run();

      Problem<3> p3d(order);
      p3d.run();
    }

  return 0;
}
