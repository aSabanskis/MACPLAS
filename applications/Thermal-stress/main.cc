#include <deal.II/base/function_parser.h>

#include <deal.II/grid/grid_in.h>

#include "../../include/stress_solver.h"

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
  initialize();

  StressSolver<dim> solver;

  FunctionParser<dim> m_T;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
{
  prm.declare_entry("Temperature",
                    "1000 + 500*z",
                    Patterns::Anything(),
                    "Temperature field in K (coordinate function)");

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

  m_T.initialize(FunctionParser<dim>::default_variable_names(),
                 prm.get("Temperature"),
                 typename FunctionParser<dim>::ConstMap());

  solver.output_parameter_table();
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  solver.solve();
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
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  std::vector<Point<dim>> support_points;
  solver.get_support_points(support_points);

  Vector<double> &temperature = solver.get_temperature();
  AssertDimension(temperature.size(), support_points.size());

  for (unsigned int i = 0; i < temperature.size(); ++i)
    {
      temperature[i] = m_T.value(support_points[i]);
    }
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init      = false;
  int  order     = 2;
  int  dimension = 3;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
      if (arguments[i] == "2d" || arguments[i] == "2D")
        dimension = 2;
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 3)
    {
      Problem<3> p3d(order, init);
      if (!init)
        p3d.run();
    }
  else if (dimension == 2)
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
