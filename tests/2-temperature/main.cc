#include <deal.II/grid/grid_generator.h>

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
  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::hyper_cube(triangulation, 0, 1, true);

  triangulation.refine_global(4);

  const unsigned int n_probes = 6;
  for (unsigned int i = 0; i < n_probes; ++i)
    {
      Point<dim> p;
      p(0) = i / (n_probes - 1.0);
      solver.add_probe(p);
    }
}

template <int dim>
void
Problem<dim>::initialize()
{
  // Physical parameters from https://doi.org/10.1016/S0022-0248(03)01253-3
  const double T0 = 1687;

  solver.initialize(); // sets T=0

  Vector<double> &temperature = solver.get_temperature();
  temperature.add(T0);

  unsigned int boundary_id = 0;
  solver.set_bc1(boundary_id, T0);

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  boundary_id = 1;
  solver.get_boundary_points(boundary_id, points, boundary_dofs);
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
  solver.set_bc_rad_mixed(boundary_id, q, emissivity, emissivity_deriv);
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}
