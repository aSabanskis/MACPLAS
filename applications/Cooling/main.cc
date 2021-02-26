#include <deal.II/base/function_parser.h>

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
  apply_temperature_bc();

  void
  solve_temperature_dislocation();

  void
  solve_temperature();

  TemperatureSolver<dim> temperature_solver;
  DislocationSolver<dim> dislocation_solver;

  constexpr static unsigned int boundary_id_top = 5;
  constexpr static unsigned int boundary_id_bot = 4;

  FunctionParser<1> m_T_top, m_T_bot;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : temperature_solver(order)
  , dislocation_solver(order)
{
  prm.declare_entry("Temperature only",
                    "false",
                    Patterns::Bool(),
                    "Calculate just the temperature field");

  prm.declare_entry(
    "Output frequency",
    "0",
    Patterns::Integer(0),
    "Number of time steps between result output (0 - disabled)");

  prm.declare_entry("Initial temperature",
                    "1685",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");

  prm.declare_entry("Top heat transfer coefficient",
                    "10",
                    Patterns::Double(0),
                    "Top heat transfer coefficient h_top in W/m^2/K");

  prm.declare_entry("Bottom heat transfer coefficient",
                    "10",
                    Patterns::Double(0),
                    "Bottom heat transfer coefficient h_bot in W/m^2/K");

  prm.declare_entry("Top reference temperature",
                    "1685 - 2 * t",
                    Patterns::Anything(),
                    "Top reference temperature T_top in K (time function)");

  prm.declare_entry("Bottom reference temperature",
                    "1685 - 5 * t",
                    Patterns::Anything(),
                    "Bottom reference temperature T_bot in K (time function)");

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

  m_T_top.initialize("t",
                     prm.get("Top reference temperature"),
                     typename FunctionParser<1>::ConstMap());

  m_T_bot.initialize("t",
                     prm.get("Bottom reference temperature"),
                     typename FunctionParser<1>::ConstMap());
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  if (prm.get_bool("Temperature only"))
    solve_temperature();
  else
    solve_temperature_dislocation();
}

template <int dim>
void
Problem<dim>::solve_temperature_dislocation()
{
  const int n_output = prm.get_integer("Output frequency");

  for (unsigned int i = 1;; ++i)
    {
      temperature_solver.get_time_step() = dislocation_solver.get_time_step();

      apply_temperature_bc();
      const bool keep_going_temp = temperature_solver.solve();

      dislocation_solver.get_temperature() =
        temperature_solver.get_temperature();

      const bool keep_going_disl = dislocation_solver.solve();

      if (!keep_going_temp || !keep_going_disl)
        break;

      if (n_output > 0 && i % n_output == 0)
        {
          temperature_solver.output_vtk();
          dislocation_solver.output_vtk();
        }
    };

  temperature_solver.output_vtk();
  dislocation_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::solve_temperature()
{
  const int n_output = prm.get_integer("Output frequency");

  for (unsigned int i = 1;; ++i)
    {
      apply_temperature_bc();
      const bool keep_going_temp = temperature_solver.solve();

      if (!keep_going_temp)
        break;

      if (n_output > 0 && i % n_output == 0)
        {
          temperature_solver.output_vtk();
        }
    };

  temperature_solver.output_vtk();
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
  temperature_solver.output_vtk();

  if (prm.get_bool("Temperature only"))
    return;

  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature;
  dislocation_solver.solve(true);
  dislocation_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::apply_temperature_bc(void)
{
  const double t =
    temperature_solver.get_time() + temperature_solver.get_time_step();

  const double h_top = prm.get_double("Top heat transfer coefficient");
  const double h_bot = prm.get_double("Bottom heat transfer coefficient");
  const double T_top = m_T_top.value(Point<1>(t));
  const double T_bot = m_T_bot.value(Point<1>(t));

  temperature_solver.set_bc_convective(boundary_id_top, h_top, T_top);
  temperature_solver.set_bc_convective(boundary_id_bot, h_bot, T_bot);

  temperature_solver.add_output("T_top[K]", T_top);
  temperature_solver.add_output("T_bot[K]", T_bot);
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
