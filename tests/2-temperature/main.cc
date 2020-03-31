#include <deal.II/base/polynomial.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include "../../include/temperature_solver.h"

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

  void
  output_results() const;

  TemperatureSolver<dim> solver;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : solver(order)
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  while (true)
    {
      const bool keep_going = solver.solve();
      solver.output_results();

      if (!keep_going)
        break;
    };
}

template <int dim>
void
Problem<dim>::make_grid()
{
  GridGenerator::hyper_cube(solver.mesh(), 0, 1, true);

  solver.mesh().refine_global(3);

  std::cout << "Number of active cells: " << solver.mesh().n_active_cells()
            << "\n";
}

template <int dim>
void
Problem<dim>::initialize()
{
  const double T0 = 1685;
  solver.initialize(T0);

  const double                          l0 = 22;
  const Polynomials::Polynomial<double> lambda(
    {l0 * 4.495, -l0 * 7.222 / T0, l0 * 3.728 / T0 / T0});
  solver.initialize(lambda);

  solver.set_bc1(0, T0);
  // solver.set_bc1(1, 1000);

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  solver.get_boundary_points(1, points, boundary_dofs);
  Vector<double> q(points.size());
  for (unsigned int i = 0; i < q.size(); ++i)
    {
      if (boundary_dofs[i])
        q[i] = 1e4;
    }
  const double                  emissivity0 = 0.46;
  std::function<double(double)> emissivity  = [=](double T) {
    const double t = T / T0;
    return emissivity0 * (t < 0.593 ? 1.39 : 1.96 - 0.96 * t);
  };
  std::function<double(double)> emissivity_deriv = [=](double T) {
    const double t = T / T0;
    return emissivity0 * (t < 0.593 ? 0 : -0.96 / T0);
  };
  solver.set_bc_rad_mixed(1, q, emissivity, emissivity_deriv);
}

int
main()
{
  Problem<1> p1d;
  p1d.run();

  Problem<2> p2d;
  p2d.run();

  Problem<3> p3d;
  p3d.run();

  return 0;
}
