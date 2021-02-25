#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_generator.h>

#include <cmath>

#include "../../include/dislocation_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order = 1, const bool use_default_prm = false);

  void
  run();

private:
  DislocationSolver<dim> solver;

  ParameterHandler prm;

  void
  make_grid();

  void
  initialize();
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
{
  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");

  prm.declare_entry("Strain rate",
                    "-1e-5", // compression
                    Patterns::Double(),
                    "Strain rate dot_epsilon in s^-1");

  prm.declare_entry("Max strain",
                    "0",
                    Patterns::Double(0),
                    "Maximum strain epsilon_max (dimensionless, 0 - disabled)");

  prm.declare_entry("L", "0.020", Patterns::Double(0), "Cube size L in m");

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
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  const double max_strain = prm.get_double("Max strain");

  while (true)
    {
      const double t      = solver.get_time() + solver.get_time_step();
      const double strain = prm.get_double("Strain rate") * t;
      const double strain_total =
        max_strain <= 0 ?
          strain :
          std::copysign(std::min(std::abs(strain), max_strain), strain);
      const double dx = prm.get_double("L") * strain_total;

      solver.get_stress_solver().set_bc1(1, 0, dx);
      solver.add_output("strain_total[-]", strain_total);

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

  GridGenerator::hyper_cube(triangulation, 0, prm.get_double("L"), true);

  // a single probe point at the origin
  solver.add_probe(Point<dim>());

  solver.add_output("strain_total[-]");
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  Vector<double> &temperature = solver.get_temperature();

  temperature = 0;
  temperature.add(prm.get_double("Initial temperature"));

  solver.get_stress_solver().set_bc1(0, 0, 0.0);
  solver.get_stress_solver().set_bc1(1, 0, 0.0);

  // initialize stresses and output probes at zero time
  solver.solve(true);
  solver.output_vtk();
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init = false;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    if (arguments[i] == "init" || arguments[i] == "use_default_prm")
      init = true;

  deallog.attach(std::cout);
  deallog.depth_console(2);

  Problem<3> p3d(1, init);
  if (!init)
    p3d.run();

  return 0;
}
