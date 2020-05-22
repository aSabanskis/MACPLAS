#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "../../include/stress_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(unsigned int order = 2);

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
Problem<dim>::Problem(unsigned int order)
  : r_a(0.5)
  , r_b(1.0)
  , T_a(500)
  , T_b{1000}
  , manifold(SphericalManifold<dim>(Point<dim>()))
  , solver(order)
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  solver.output_results();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Assert(r_a > 0, ExcMessage("r_a must be positive"));
  Assert(r_b > r_a, ExcMessage("r_b must be larger than r_b"));

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::hyper_shell(triangulation, Point<dim>(), r_a, r_b, 0, true);

  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold(0, manifold);
  triangulation.set_manifold(1, manifold);

  triangulation.refine_global(3);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << "\n";
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  std::vector<Point<dim>> support_points;
  solver.get_support_points(support_points);

  Vector<double> &temperature = solver.get_temperature();
  // Set the temperature using analytical solution in spherical coordinates.
  const double c1 = r_a * (T_a - T_b) / (1 - r_a / r_b);
  const double c2 = (T_b - T_a * r_a / r_b) / (1 - r_a / r_b);
  for (unsigned int i = 0; i < support_points.size(); ++i)
    {
      temperature[i] = c1 / support_points[i].norm() + c2;
    }
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}
