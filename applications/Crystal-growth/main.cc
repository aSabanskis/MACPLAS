#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <algorithm>
#include <fstream>

#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order = 2, const bool use_default_prm = false);

  void
  run();

private:
  void
  make_grid();

  void
  deform_grid();

  void
  initialize_temperature();

  TemperatureSolver<dim> temperature_solver;

  Triangulation<dim> triangulation;

  // crystallization interface
  constexpr static unsigned int boundary_id_interface = 0;
  // crystal side surface
  constexpr static unsigned int boundary_id_surface = 1;
  // crystal axis
  constexpr static unsigned int boundary_id_axis = 2;

  unsigned int point_id_axis_z_min, point_id_axis_z_max, point_id_triple;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : temperature_solver(order, use_default_prm)
{
  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize_temperature();

  // proof of concept
  while (true)
    {
      deform_grid();

      bool keep_going_temp = temperature_solver.solve();

      temperature_solver.output_vtk();

      if (!keep_going_temp)
        break;
    };
}

template <int dim>
void
Problem<dim>::make_grid()
{
  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh-" + std::to_string(dim) + "d.msh");
  gi.read_msh(f);

  temperature_solver.get_mesh().copy_triangulation(triangulation);

  // print info
  std::cout << "Number of cells: " << triangulation.n_cells() << "\n"
            << "Number of points: " << triangulation.n_vertices() << "\n";

  const auto bounds = triangulation.get_boundary_ids();
  for (const auto &b : bounds)
    {
      const auto points = get_boundary_points(triangulation, b);
      std::cout << "Boundary No. " << (int)b << " : " << points.size()
                << " points\n";
    }

  // preprocess some point indices
  auto cmp_z = [](const auto &it1, const auto &it2) {
    return it1.second[dim - 1] < it2.second[dim - 1];
  };

  const auto points_axis = get_boundary_points(triangulation, boundary_id_axis),
             points_surface =
               get_boundary_points(triangulation, boundary_id_surface);
  AssertThrow(!points_axis.empty(),
              ExcMessage("No points on boundary No. " +
                         std::to_string(boundary_id_axis) + " (axis) found"));
  AssertThrow(!points_surface.empty(),
              ExcMessage("No points on boundary No. " +
                         std::to_string(boundary_id_surface) +
                         " (side surface) found"));

  point_id_axis_z_min =
    std::min_element(points_axis.begin(), points_axis.end(), cmp_z)->first;
  point_id_axis_z_max =
    std::max_element(points_axis.begin(), points_axis.end(), cmp_z)->first;
  point_id_triple =
    std::min_element(points_surface.begin(), points_surface.end(), cmp_z)
      ->first;

  std::cout << "Axis lowest point = " << point_id_axis_z_min << '\n'
            << "Axis highest point = " << point_id_axis_z_max << '\n'
            << "Triple point = " << point_id_triple << '\n';
}

template <int dim>
void
Problem<dim>::deform_grid()
{
  const auto points_axis = get_boundary_points(triangulation, boundary_id_axis),
             points_surface =
               get_boundary_points(triangulation, boundary_id_surface),
             points_interface =
               get_boundary_points(triangulation, boundary_id_interface);

  auto shift_point = [](const Point<dim> &p) {
    Point<dim> p_new = p;
    // TODO: remove hardcoded values
    p_new[0] += 0.1e-3 * std::sin(p[0] * 200);
    p_new[dim - 1] += -1e-3 * std::cos(p[0] * 50);
    return p_new;
  };

  const Point<dim> p1 = points_axis.at(point_id_axis_z_min),
                   p2 = points_axis.at(point_id_axis_z_max);
  const auto dp_axis  = shift_point(p1) - p1;
#ifdef DEBUG
  std::cout << "p_axis = " << p1 << " dp_axis = " << dp_axis << '\n';
#endif

  auto points_new = points_axis;
  for (auto &it : points_new)
    {
      it.second += dp_axis * (p2 - it.second).norm() / (p2 - p1).norm();
    }

  const Point<dim> p_triple  = points_surface.at(point_id_triple);
  const auto       dp_triple = shift_point(p_triple) - p_triple;
#ifdef DEBUG
  std::cout << "p_triple = " << p_triple << " dp_triple = " << dp_triple
            << '\n';
#endif

  for (const auto &it : points_surface)
    {
      // TODO: use a better approach to preserve the initial crystal shape
      points_new[it.first] = it.second + dp_triple * (p2 - it.second).norm() /
                                           (p2 - dp_triple).norm();
    }

  for (const auto &it : points_interface)
    {
      points_new[it.first] = shift_point(it.second);
    }

  // update_boundary_points(triangulation, points_new);

  GridTools::laplace_transform(points_new, triangulation);

  temperature_solver.get_mesh().clear();
  temperature_solver.get_mesh().copy_triangulation(triangulation);
}

template <int dim>
void
Problem<dim>::initialize_temperature()
{
  temperature_solver.initialize(); // sets T=0

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Initial temperature"));

  temperature_solver.output_mesh();
  temperature_solver.output_parameter_table();

  temperature_solver.output_vtk();
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init      = false;
  int  order     = 2;
  int  dimension = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 2)
    {
      Problem<2> p2d(order, init);
      if (!init)
        p2d.run();
    }
  else
    {
      throw ExcNotImplemented();
    }

  return 0;
}
