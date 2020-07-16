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

  prm.declare_entry("Initial dislocation density",
                    "1e3",
                    Patterns::Double(),
                    "Initial dislocation density in m^-2");

  prm.declare_entry("Displacement",
                    "1e-7",
                    Patterns::Double(),
                    "Displacement in m");

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

  GridGenerator::hyper_cube(triangulation, 0, 0.02, true);

  // a single probe point at the origin
  solver.add_probe(Point<dim>());
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  Vector<double> &temperature         = solver.get_temperature();
  Vector<double> &dislocation_density = solver.get_dislocation_density();

  temperature = 0;
  temperature.add(prm.get_double("Initial temperature"));

  dislocation_density = 0;
  dislocation_density.add(prm.get_double("Initial dislocation density"));

  solver.get_stress_solver().set_bc1(0, 0, 0.0);
  solver.get_stress_solver().set_bc1(1, 0, prm.get_double("Displacement"));
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}
