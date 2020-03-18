#include "../../include/temperature_solver.h"

#include <deal.II/base/polynomial.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

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
  output_results();
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

  const double l0 = 22;
  const Polynomials::Polynomial<double> lambda({l0*4.495, -l0*7.222/T0, l0*3.728/T0/T0});
  solver.initialize(lambda);
}

template <int dim>
void
Problem<dim>::output_results() const
{
  solver.output_results();
}

int
main()
{
  Problem<2> p2d;
  p2d.run();

  Problem<3> p3d;
  p3d.run();

  return 0;
}
