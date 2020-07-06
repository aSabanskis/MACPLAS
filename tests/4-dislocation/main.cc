#include <deal.II/grid/grid_generator.h>

#include "../../include/dislocation_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order = 1);

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
Problem<dim>::Problem(const unsigned int order)
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
  Triangulation<dim> &triangulation = solver.get_mesh();

  GridGenerator::hyper_cube(triangulation, 0, 0.02, true);

  // a single probe point at the origin
  solver.add_probe(Point<dim>());
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

  solver.get_stress_solver().set_bc1(0, 0, 0.0);
  solver.get_stress_solver().set_bc1(1, 0, 1e-7);
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}