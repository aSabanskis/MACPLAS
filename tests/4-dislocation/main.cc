#include <deal.II/base/parameter_handler.h>

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

  ParameterHandler prm;

  void
  make_grid();

  void
  initialize();
};

template <int dim>
Problem<dim>::Problem(const unsigned int order)
  : solver(order)
{
  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(),
                    "Initial temperature in K");

  prm.declare_entry("Strain rate",
                    "1e-5",
                    Patterns::Double(),
                    "Strain rate in s^-1");

  prm.declare_entry("L", "0.020", Patterns::Double(0), "Cube size in m");

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

  while (true)
    {
      const double t  = solver.get_time() + solver.get_time_step();
      const double dx = prm.get_double("L") * prm.get_double("Strain rate") * t;
      solver.get_stress_solver().set_bc1(1, 0, -dx); // compression

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
main()
{
  deallog.attach(std::cout);
  deallog.depth_console(2);

  Problem<3> p3d;
  p3d.run();

  return 0;
}
