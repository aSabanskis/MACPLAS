#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <cmath>

#include "../../include/dislocation_solver.h"

using namespace dealii;

enum class BCType
{
  three_point_load,
  uniaxial_compression_force,
  uniaxial_compression_displacement
};

template <int dim>
class Problem
{
public:
  explicit Problem(const unsigned int order,
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

  // helper functions
  static bool
  cmp_x_pair(const std::pair<unsigned int, Point<dim>> &it1,
             const std::pair<unsigned int, Point<dim>> &it2);

  static bool
  cmp_z_pair(const std::pair<unsigned int, Point<dim>> &it1,
             const std::pair<unsigned int, Point<dim>> &it2);

  static bool
  cmp_z(const Point<dim> &p1, const Point<dim> &p2);


  Timer timer;

  void
  initialize();

  void
  update_BCs(const double time);

  void
  apply_force(const double time);

  void
  apply_displacement(const double time);

  /** Checks whether the results need to be outputted at the end of the current
   * time step, and adjusts it if necessary */
  bool
  update_output_time_step();

  void
  set_time_step(const double dt);

  constexpr static unsigned int boundary_id_free = 0;
  constexpr static unsigned int boundary_id_load = 1;
  constexpr static unsigned int boundary_id_wall = 2;

  BCType bc_type;

  double load_area;

  int load_component;

  FunctionParser<1> displacement;

  double previous_time_step;
  double next_output_time;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
  , bc_type(BCType::three_point_load)
  , load_area(0)
  , load_component(0)
  , previous_time_step()
  , next_output_time()
{
  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");

  prm.declare_entry("Max force",
                    "0",
                    Patterns::Double(0),
                    "Maximum applied force in N");

  prm.declare_entry(
    "Force ramp",
    "0",
    Patterns::Double(0),
    "Time over which force reaches max value in s (0 - instantaneous)");

  prm.declare_entry("Displacement x",
                    "1e-6 * t",
                    Patterns::Anything(),
                    "Horizontal displacement in m (time function)");

  prm.declare_entry("Max dx",
                    "0",
                    Patterns::Double(0),
                    "Maximum horizontal displacement in m (0 - disabled)");

  prm.declare_entry("Max dz",
                    "0",
                    Patterns::Double(0),
                    "Maximum vertical displacement in m (0 - disabled)");

  prm.declare_entry(
    "Load type",
    "Three point load",
    Patterns::Selection(
      "Three point load|Uniaxial compression force|Uniaxial compression displacement"),
    "Type of the boundary conditions");

  prm.declare_entry("Load x",
                    "0.002",
                    Patterns::Double(0),
                    "Position of the support (between -x and +x)");

  prm.declare_entry("Support x",
                    "0.03",
                    Patterns::Double(0),
                    "Position of the support (located at +x and -x)");

  prm.declare_entry("Output time step",
                    "60",
                    Patterns::Double(0),
                    "Time interval in s between result output (0 - disabled). "
                    "The time step is adjusted to match the required time.");

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

  const double dx_max = prm.get_double("Max dx");
  const double dz_max = prm.get_double("Max dz");

  while (true)
    {
      const bool output_enabled = update_output_time_step();

      const double t = solver.get_time() + solver.get_time_step();

      update_BCs(t);

      const auto &dx = solver.get_displacement().block(0);
      const auto &dz = solver.get_displacement().block(dim - 1);

      const auto limits_dx = minmax(dx);
      const auto limits_dz = minmax(dz);

      const bool dx_reached =
        dx_max > 0 && (std::abs(limits_dx.first) >= dx_max ||
                       std::abs(limits_dx.second) >= dx_max);

      const bool dz_reached =
        dz_max > 0 && (std::abs(limits_dz.first) >= dz_max ||
                       std::abs(limits_dz.second) >= dz_max);

      const bool keep_going = solver.solve() && !dx_reached && !dz_reached;

      if (!keep_going)
        break;

      if (output_enabled)
        {
          solver.output_vtk();

          std::cout << "Restoring previous dt=" << previous_time_step << "s\n";
          set_time_step(previous_time_step);
        }
    };

  solver.output_vtk();

  std::cout << "Finished in " << timer.wall_time() << " s\n";
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
}

template <int dim>
void
Problem<dim>::handle_boundaries()
{
  Triangulation<dim> &triangulation = solver.get_mesh();

  std::map<unsigned int, unsigned int> boundary_info =
    get_boundary_summary(triangulation);

  const std::string lt = prm.get("Load type");

  if (lt == "Three point load")
    bc_type = BCType::three_point_load;
  else if (lt == "Uniaxial compression force")
    bc_type = BCType::uniaxial_compression_force;
  else if (lt == "Uniaxial compression displacement")
    bc_type = BCType::uniaxial_compression_displacement;

  const unsigned int n_expected = bc_type == BCType::three_point_load ? 2 : 3;

  if (boundary_info.size() < n_expected)
    {
      std::cout << boundary_info.size() << " boundary/ies detected instead of "
                << n_expected << ", setting custom boundary IDs\n";

      const std::map<unsigned int, Point<dim>> points0 =
        get_boundary_points(triangulation, boundary_info.begin()->first);

      const double x_min =
        std::min_element(points0.begin(), points0.end(), cmp_x_pair)->second[0];

      const double x_max =
        std::max_element(points0.begin(), points0.end(), cmp_x_pair)->second[0];

      const double z_max =
        std::max_element(points0.begin(), points0.end(), cmp_z_pair)
          ->second[dim - 1];

      std::cout << "x_min: " << x_min << " x_max: " << x_max
                << " z_max: " << z_max << '\n';

      const double x_load = prm.get_double("Load x");

      typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

      for (; cell != endc; ++cell)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                // to avoid repeating it afterwards
                cell->face(f)->set_boundary_id(boundary_id_free);

                const Point<dim> face_center = cell->face(f)->center();

                const bool is_left  = face_center[0] <= x_min;
                const bool is_right = face_center[0] >= x_max;
                const bool is_top   = face_center[dim - 1] >= z_max;

                if (bc_type == BCType::three_point_load)
                  {
                    if (is_top && std::abs(face_center[0]) <= x_load)
                      {
                        cell->face(f)->set_boundary_id(boundary_id_load);
                        load_area += cell->face(f)->measure();
                      }
                  }
                else
                  {
                    if (is_right)
                      {
                        cell->face(f)->set_boundary_id(boundary_id_load);
                        load_area += cell->face(f)->measure();
                      }
                    else if (is_left)
                      {
                        cell->face(f)->set_boundary_id(boundary_id_wall);
                      }
                  }
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
Problem<dim>::cmp_x_pair(const std::pair<unsigned int, Point<dim>> &it1,
                         const std::pair<unsigned int, Point<dim>> &it2)
{
  return it1.second[0] < it2.second[0];
}

template <int dim>
bool
Problem<dim>::cmp_z_pair(const std::pair<unsigned int, Point<dim>> &it1,
                         const std::pair<unsigned int, Point<dim>> &it2)
{
  return it1.second[dim - 1] < it2.second[dim - 1];
}


template <int dim>
bool
Problem<dim>::cmp_z(const Point<dim> &p1, const Point<dim> &p2)
{
  return p1[dim - 1] < p2[dim - 1];
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  Vector<double> &temperature = solver.get_temperature();

  temperature = 0;
  temperature.add(prm.get_double("Initial temperature"));

  next_output_time = prm.get_double("Output time step");

  if (bc_type == BCType::three_point_load)
    {
      load_component = dim - 1;

      std::vector<Point<dim>> points0;
      solver.get_support_points(points0);

      const double z_min =
        (*std::min_element(points0.begin(), points0.end(), cmp_z))[dim - 1];

      const double x_support = prm.get_double("Support x");

      for (size_t i = 0; i < points0.size(); ++i)
        {
          const auto &p = points0[i];

          const bool is_bot = p[dim - 1] <= z_min;

          if (is_bot && (std::abs(p[0] - x_support) <= 1e-8 ||
                         std::abs(p[0] + x_support) <= 1e-8))
            solver.get_stress_solver().set_bc1_dof(i, dim - 1, 0.0);
        }
    }
  else
    {
      load_component = 0;
      solver.get_stress_solver().set_bc1(boundary_id_wall, 0, 0.0);
    }

  if (bc_type == BCType::uniaxial_compression_displacement)
    displacement.initialize("t",
                            prm.get("Displacement x"),
                            typename FunctionParser<1>::ConstMap());


  std::cout << "Load area: " << load_area << " m2\n";
  std::cout << "Max pressure: " << prm.get_double("Max force") / load_area
            << " Pa\n";

  // initialize stresses and output probes at zero time
  update_BCs(0);
  solver.solve(true);
  solver.output_vtk();
}

template <int dim>
void
Problem<dim>::update_BCs(const double time)
{
  if (bc_type == BCType::uniaxial_compression_displacement)
    apply_displacement(time);
  else
    apply_force(time);
}

template <int dim>
void
Problem<dim>::apply_force(const double time)
{
  const double F0   = prm.get_double("Max force");
  const double ramp = prm.get_double("Force ramp");

  const double F_scale = ramp <= 0 ? 1 : time >= ramp ? 1 : time / ramp;
  const double F       = F0 * F_scale;

  Tensor<1, dim> load;
  load[load_component] = -F / load_area;

  solver.get_stress_solver().set_bc_load(boundary_id_load, load);
  solver.add_output("force[N]", F);
}

template <int dim>
void
Problem<dim>::apply_displacement(const double time)
{
  const double dx = displacement.value(Point<1>(time));
  solver.get_stress_solver().set_bc1(boundary_id_load, load_component, -dx);
  solver.add_output("dx[m]", dx);
}

template <int dim>
bool
Problem<dim>::update_output_time_step()
{
  const double t  = solver.get_time();
  const double dt = solver.get_time_step();

  const double dt_output = prm.get_double("Output time step");

  previous_time_step = dt;

  // output time step is not specified
  if (dt_output <= 0)
    return false;

  // output time too far away, nothing to do
  if (dt_output > 0 && t + (1 + 1e-4) * dt < next_output_time)
    return false;

  // output time is specified and the time step has to be adjusted
  set_time_step(next_output_time - t);

  std::cout << "dt changed from " << dt << " to " << solver.get_time_step()
            << " s for result output at t=" << next_output_time << " s\n";

  next_output_time += dt_output;

  return true;
}

template <int dim>
void
Problem<dim>::set_time_step(const double dt)
{
  solver.get_time_step() = dt;
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
