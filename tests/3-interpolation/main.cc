#include <deal.II/base/polynomial.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

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

  void
  output_results() const;

  CylindricalManifold<dim> manifold;

  TemperatureSolver<dim> solver;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : manifold(2)
  , solver(order)
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
  Assert(dim == 3, ExcNotImplemented());

  Triangulation<dim - 1> base;
  GridGenerator::hyper_ball(base, Point<dim - 1>(), 0.1);

  GridGenerator::extrude_triangulation(base, 5, 0.5, solver.mesh());

  solver.mesh().set_all_manifold_ids(0);
  solver.mesh().set_manifold(0, manifold);
  solver.mesh().refine_global(3);

  std::cout << "Number of active cells: " << solver.mesh().n_active_cells()
            << "\n";
}

template <int dim>
void
Problem<dim>::initialize()
{
  const double T0 = 300;
  solver.initialize(T0);

  const Polynomials::Polynomial<double> lambda(std::vector<double>({1.0}));
  solver.initialize(lambda);

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  const unsigned int      boundary_id = 0;
  solver.get_boundary_points(boundary_id, points, boundary_dofs);
  Vector<double> q(points.size());

  SurfaceInterpolator3D surf;
  surf.read_vtk("q.vtk");
  surf.convert(SurfaceInterpolator3D::CellField,
               "q",
               SurfaceInterpolator3D::PointField,
               "q_from_cell");
  surf.convert(SurfaceInterpolator3D::PointField,
               "q",
               SurfaceInterpolator3D::CellField,
               "q_from_point");
  surf.write_vtu("q.vtu");
  surf.interpolate(
    SurfaceInterpolator3D::PointField, "q", points, boundary_dofs, q);

  std::function<double(double)> zero = [=](double T) { return 0; };
  solver.set_bc_rad_mixed(boundary_id, q, zero, zero);
}

int
main()
{
  Problem<3> p3d(2);
  p3d.run();

  return 0;
}
