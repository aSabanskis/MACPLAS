#include <deal.II/grid/grid_in.h>

#include <fstream>

#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

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

      if (!keep_going)
        break;
    };

  solver.output_results();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = solver.get_mesh();

  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh.msh");
  gi.read_msh(f);
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize(); // sets T=0

  Vector<double> &temperature = solver.get_temperature();
  temperature.add(1000);

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  unsigned int            boundary_id = 0;
  solver.get_boundary_points(boundary_id, points, boundary_dofs);
  Vector<double> q(points.size());

  SurfaceInterpolator3D surf;
  surf.read_vtu("qEM.vtu");
  surf.convert(SurfaceInterpolator3D::CellField,
               "QEM",
               SurfaceInterpolator3D::PointField,
               "q");
  surf.interpolate(
    SurfaceInterpolator3D::PointField, "q", points, boundary_dofs, q);

  // scale by the square of current
  q *= 1e-6 * sqr(105);

  // Physical parameters from https://doi.org/10.1016/j.jcrysgro.2020.125842
  std::function<double(double)> emissivity       = [](double) { return 0.57; };
  std::function<double(double)> emissivity_deriv = [](double) { return 0.0; };

  solver.set_bc_rad_mixed(boundary_id, q, emissivity, emissivity_deriv);
}

int
main()
{
  Problem<3> p3d(2);
  p3d.run();

  return 0;
}
