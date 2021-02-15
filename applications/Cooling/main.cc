#include <deal.II/grid/grid_generator.h>

#include "../../include/dislocation_solver.h"
#include "../../include/temperature_solver.h"

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
  apply_temperature_bc(void);


  TemperatureSolver<dim> temperature_solver;
  DislocationSolver<dim> dislocation_solver;

  constexpr static double T0 = 1685;

  constexpr static unsigned int boundary_id_top = 5;
  constexpr static unsigned int boundary_id_bot = 4;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : temperature_solver(order)
  , dislocation_solver(order)
{
  prm.declare_entry("Initial temperature",
                    std::to_string(T0),
                    Patterns::Double(0),
                    "Initial temperature in K");

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
      temperature_solver.get_time_step() = dislocation_solver.get_time_step();

      apply_temperature_bc();
      const bool keep_going_temp = temperature_solver.solve();

      dislocation_solver.get_temperature() =
        temperature_solver.get_temperature();

      const bool keep_going_disl = dislocation_solver.solve();

      if (!keep_going_temp || !keep_going_disl)
        break;
    };

  temperature_solver.output_vtk();
  dislocation_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = temperature_solver.get_mesh();

  const Point<dim> p(-0.42, 0.42, -0.20);

  GridGenerator::subdivided_hyper_rectangle(
    triangulation, {21, 21, 10}, p, -p, true);

  dislocation_solver.get_mesh().copy_triangulation(triangulation);
}

template <int dim>
void
Problem<dim>::initialize()
{
  temperature_solver.initialize(); // sets T=0

  temperature_solver.output_mesh();

  for (int i = -1; i <= 1; ++i)
    {
      Point<dim> p;
      p[dim - 1] = 0.2 * i;
      temperature_solver.add_probe(p);
      dislocation_solver.add_probe(p);
    }

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Initial temperature"));

  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature;
  dislocation_solver.solve(true);

  temperature_solver.output_vtk();
  dislocation_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::apply_temperature_bc(void)
{
  const double t =
    temperature_solver.get_time() + temperature_solver.get_time_step();

  temperature_solver.set_bc1(boundary_id_top, T0 - 2 * t);
  temperature_solver.set_bc1(boundary_id_bot, T0 - 5 * t);
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
