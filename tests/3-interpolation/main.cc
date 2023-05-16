#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  explicit Problem(const unsigned int order = 2);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  CylindricalManifold<dim> manifold;

  TemperatureSolver<dim>     solver;
  TemperatureSolver<dim - 1> solver2;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order)
  : manifold(2)
  , solver(order)
  , solver2(order)
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();
  // do not calculate
  solver.output_vtk();
  solver2.output_vtk();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Assert(dim == 3, ExcNotImplemented());

  Triangulation<dim - 1> base;
  GridGenerator::hyper_ball(base, Point<dim - 1>(), 0.1);

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::extrude_triangulation(base, 5, 0.5, triangulation);

  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold(0, manifold);
  triangulation.refine_global(3);

  typename Triangulation<dim>::active_cell_iterator cell = triangulation
                                                             .begin_active(),
                                                    endc = triangulation.end();
  for (; cell != endc; ++cell)
    {
      for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
        {
          auto face = cell->face(i);
          if (face->at_boundary() && face->boundary_id() == 0 &&
              face->center()[1] < 0)
            face->set_boundary_id(3);
        }
    }

  Triangulation<dim - 1> &triangulation2 = solver2.get_mesh();
  GridGenerator::hyper_rectangle(triangulation2,
                                 Point<dim - 1>(-0.1, 0),
                                 Point<dim - 1>(0.1, 0.5));
  triangulation2.refine_global(3);
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();
  solver2.initialize();
  // no need to initialize the temperature field

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  unsigned int            boundary_id = 0;
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

  std::function<double(const double)> zero = [=](const double) { return 0; };
  solver.set_bc_rad_mixed(boundary_id, q, zero, zero);


  SurfaceInterpolator2D surf2;
  surf2.read_txt("q-2d.txt");

  boundary_id = 3;
  solver.get_boundary_points(boundary_id, points, boundary_dofs);
  surf2.interpolate("q", points, boundary_dofs, q);
  solver.set_bc_rad_mixed(boundary_id, q, zero, zero);

  std::vector<Point<dim - 1>> points2;
  boundary_id = 0;
  solver2.get_boundary_points(boundary_id, points2, boundary_dofs);
  // test all DOFs, not just boundary
  std::fill(boundary_dofs.begin(), boundary_dofs.end(), true);
  q.reinit(points2.size());
  surf.interpolate(
    SurfaceInterpolator3D::PointField, "q", points2, boundary_dofs, q);
  solver2.set_bc_rad_mixed(boundary_id, q, zero, zero);
}

int
main()
{
  deallog.attach(std::cout);
  deallog.depth_console(2);

  Problem<3> p3d(2);
  p3d.run();

  return 0;
}
