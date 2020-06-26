#include <deal.II/grid/grid_generator.h>

#include "../../include/dislocation_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem();

  void
  run();

private:
  DislocationSolver<dim> solver;

  void
  make_grid();

  void
  initialize();
};

template <int dim>
Problem<dim>::Problem()
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  // do not calculate
  solver.output_results();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = solver.get_mesh();

  GridGenerator::hyper_cube(triangulation, 0, 1, true);
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  Vector<double> &temperature         = solver.get_temperature();
  Vector<double> &dislocation_density = solver.get_dislocation_density();

  for (unsigned int i = 0; i < temperature.size(); ++i)
    {
      temperature[i]         = 1000;
      dislocation_density[i] = 1e3;
    }
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}
