#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <cmath>

#include "../../include/dislocation_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  explicit Problem(const unsigned int order           = 1,
                   const bool         use_default_prm = false);

  void
  run();

private:
  DislocationSolver<dim> solver;

  ParameterHandler prm;

  void
  make_grid();

  void
  handle_boundaries();

  // helper function
  static bool
  cmp_z(const std::pair<unsigned int, Point<dim>> &it1,
        const std::pair<unsigned int, Point<dim>> &it2);

  void
  initialize();

  constexpr static unsigned int boundary_id_free    = 0;
  constexpr static unsigned int boundary_id_support = 1;
  constexpr static unsigned int boundary_id_load    = 2;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
{
  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");

  prm.declare_entry("Pressure",
                    "0",
                    Patterns::Double(0),
                    "Maximum applied pressure in Pa");

  prm.declare_entry(
    "Pressure ramp",
    "0",
    Patterns::Double(0),
    "Time over which pressure reaches max value in s (0 - instantaneous)");

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
  prm.print_parameters(std::cout, ParameterHandler::Text);
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  const double p0   = prm.get_double("Pressure");
  const double ramp = prm.get_double("Pressure ramp");

  while (true)
    {
      const double t       = solver.get_time() + solver.get_time_step();
      const double p_scale = ramp <= 0 ? 1 : t >= ramp ? 1 : t / ramp;
      const double p       = p0 * p_scale;

      Tensor<1, dim> load;
      load[dim - 1] = -p;

      solver.get_stress_solver().set_bc_load(boundary_id_load, load);
      solver.add_output("pressure[Pa]", p);

      const bool keep_going = solver.solve();

      if (!keep_going)
        break;
    };

  solver.output_vtk();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = solver.get_mesh();

  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh-" + std::to_string(dim) + "d.msh");
  gi.read_msh(f);

  handle_boundaries();

  // a single probe point at the origin
  solver.add_probe(Point<dim>());

  solver.add_output("pressure[Pa]");
}

template <int dim>
void
Problem<dim>::handle_boundaries()
{
  Triangulation<dim> &triangulation = solver.get_mesh();

  std::map<unsigned int, unsigned int> boundary_info =
    get_boundary_summary(triangulation);

  if (boundary_info.size() < 3)
    {
      std::cout << boundary_info.size()
                << " boundary/ies detected, setting custom boundary IDs\n";

      const std::map<unsigned int, Point<dim>> points0 =
        get_boundary_points(triangulation, boundary_info.begin()->first);

      const double z_min =
        std::min_element(points0.begin(), points0.end(), cmp_z)
          ->second[dim - 1];
      const double z_max =
        std::max_element(points0.begin(), points0.end(), cmp_z)
          ->second[dim - 1];

      typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

      for (; cell != endc; ++cell)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                const Point<dim> face_center = cell->face(f)->center();

                const bool is_bot = face_center[dim - 1] <= z_min;
                const bool is_top = face_center[dim - 1] >= z_max;

                if (is_top && std::abs(face_center[0]) <= 0.001)
                  cell->face(f)->set_boundary_id(boundary_id_load);
                else if (is_bot && (std::abs(face_center[0] - 0.03) <= 0.001 ||
                                    std::abs(face_center[0] + 0.03) <= 0.001))
                  cell->face(f)->set_boundary_id(boundary_id_support);
                else
                  cell->face(f)->set_boundary_id(boundary_id_free);
              }
          }

      GridOutFlags::Msh flags(true);

      GridOut go;
      go.set_flags(flags);

      std::ofstream f_out("mesh-" + std::to_string(dim) + "d-processed.msh");
      f_out << std::setprecision(16);
      go.write_msh(triangulation, f_out);

      boundary_info = get_boundary_summary(triangulation);
    }

  for (const auto &it : boundary_info)
    std::cout << "boundary " << it.first << " size: " << it.second << '\n';
}

template <int dim>
bool
Problem<dim>::cmp_z(const std::pair<unsigned int, Point<dim>> &it1,
                    const std::pair<unsigned int, Point<dim>> &it2)
{
  return it1.second[dim - 1] < it2.second[dim - 1];
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  Vector<double> &temperature = solver.get_temperature();

  temperature = 0;
  temperature.add(prm.get_double("Initial temperature"));

  solver.get_stress_solver().set_bc1(boundary_id_support, dim - 1, 0.0);

  // initialize stresses and output probes at zero time
  solver.solve(true);
  solver.output_vtk();
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init  = false;
  int  order = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  Problem<3> p3d(order, init);
  if (!init)
    p3d.run();

  return 0;
}
