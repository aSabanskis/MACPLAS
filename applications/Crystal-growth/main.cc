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

  SurfaceInterpolator2D surface_projector;

  std::unique_ptr<Function<1>> pull_rate;

  std::unique_ptr<Function<1>> crystal_radius;

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

  prm.declare_entry("Crystal radius",
                    "0.01 + 0.5e-3 * sin(t * 0.5)",
                    Patterns::Anything(),
                    "Crystal radius R in m (time function or data file name)");

  prm.declare_entry(
    "Pull rate",
    "1e-5",
    Patterns::Anything(),
    "Crystal pull rate V in m/s (time function or data file name)");

  if (use_default_prm)
    {
      std::ofstream of("problem.prm");
      prm.print_parameters(of, ParameterHandler::Text);
    }
  else
    try
      {
        prm.parse_input("problem.prm");
      }
    catch (std::exception &e)
      {
        std::cout << e.what() << "\n";

        std::ofstream of("problem-default.prm");
        prm.print_parameters(of, ParameterHandler::Text);
      }

  // print all problem-specific and solver parameters
  std::cout << "# ---------------------\n"
            << "# Problem\n";
  prm.print_parameters(std::cout, ParameterHandler::Text);

  std::cout << "# ---------------------\n"
            << "# " << temperature_solver.solver_name() << '\n';
  temperature_solver.get_parameters().print_parameters(std::cout,
                                                       ParameterHandler::Text);

  initialize_function(crystal_radius, prm.get("Crystal radius"));

  initialize_function(pull_rate, prm.get("Pull rate"));
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

  // save the crystal side surface
  std::vector<Point<dim>> points_sorted;
  for (const auto &it : points_surface)
    {
      points_sorted.push_back(it.second);
    }
  std::sort(points_sorted.begin(),
            points_sorted.end(),
            [](const auto &p1, const auto &p2) {
              // from highest to lowest, so that additional points can be added
              return p1[dim - 1] > p2[dim - 1];
            });
#ifdef DEBUG
  std::cout << "points_sorted.front = " << points_sorted.front()
            << " points_sorted.back = " << points_sorted.back() << '\n';
#endif
  surface_projector.set_points(points_sorted);

  const double R0 = crystal_radius->value(Point<1>());
  const double R  = points_surface.at(point_id_triple)[0];

  AssertThrow(std::fabs(R - R0) < 1e-4,
              ExcMessage("The actual radius R = " + std::to_string(R) +
                         " m differs from the expected R0 = " +
                         std::to_string(R0) + " m"));

  temperature_solver.add_output("R[m]", R);
}

template <int dim>
void
Problem<dim>::deform_grid()
{
  const auto &points = triangulation.get_vertices();

  const auto points_axis = get_boundary_points(triangulation, boundary_id_axis),
             points_surface =
               get_boundary_points(triangulation, boundary_id_surface),
             points_interface =
               get_boundary_points(triangulation, boundary_id_interface);

  const Point<dim> p_axis_1 = points.at(point_id_axis_z_min),
                   p_axis_2 = points.at(point_id_axis_z_max),
                   p_triple = points.at(point_id_triple);

  const double t  = temperature_solver.get_time();
  const double dt = temperature_solver.get_time_step();
  const double R0 = crystal_radius->value(Point<1>(t));
  const double R  = p_triple[0];
  const double V  = pull_rate->value(Point<1>(t));

  temperature_solver.add_output("R[m]", R0);
  temperature_solver.add_output("V[m/s]", V);

  auto calc_interface_displacement = [=](const Point<dim> &p) {
    Point<dim> dp;

    // TODO: remove hardcoded values
    dp[0]       = (R0 - R) * (p[0] / R);
    dp[dim - 1] = -1e-4 * std::cos(p[0] * 50) * std::sin(t * 0.3) - p[dim - 1];

    return dp;
  };

  const auto dp_axis   = calc_interface_displacement(p_axis_1);
  const auto dp_triple = calc_interface_displacement(p_triple);

#ifdef DEBUG
  std::cout << "p_axis = " << p_axis_1 << " dp_axis = " << dp_axis << '\n'
            << "p_triple = " << p_triple << " dp_triple = " << dp_triple
            << '\n';
#endif

  auto points_new = points_axis;
  for (auto &it : points_new)
    {
      const auto dp =
        dp_axis * (p_axis_2 - it.second).norm() / (p_axis_2 - p_axis_1).norm();

      it.second += dp;
    }

  for (const auto &it : points_surface)
    {
      const auto dp = dp_triple * (p_axis_2 - it.second)[dim - 1] /
                      (p_axis_2 - p_triple)[dim - 1];

      const auto p = it.second + dp;

      points_new[it.first] = surface_projector.project(p);
    }

  for (const auto &it : points_interface)
    {
      const auto dp = calc_interface_displacement(it.second);

      points_new[it.first] = it.second + dp;
    }

  GridTools::laplace_transform(points_new, triangulation);

  // shift the whole mesh according to the pull rate
  Point<dim> dz;
  dz[dim - 1] = V * dt;
  GridTools::shift(dz, triangulation);

  // update the precise crystal surface shape
  const auto p_triple_new     = p_triple + dp_triple;
  const auto p_triple_old     = surface_projector.get_points().back();
  auto       projector_points = surface_projector.get_points();

  // extend the surface by adding a new point
  if ((p_triple_new - p_triple_old).norm() > 0.1e-3)
    projector_points.push_back(p_triple_new);

  // shift all points
  for (auto &p : projector_points)
    p += dz;

  surface_projector.set_points(projector_points);

#ifdef DEBUG
  std::stringstream ss;
  ss << std::setprecision(8) << "crystal-surface-t" << t << ".dat";
  std::ofstream out(ss.str());
  for (const auto &p : projector_points)
    out << p << '\n';
#endif

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
