#include <deal.II/grid/grid_generator.h>

#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order, const bool use_default_prm = false);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  void
  calculate_f();

  void
  smooth();

  TemperatureSolver<dim> solver;

  BlockVector<double> function_values;

  std::vector<std::string> functions;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
{
  prm.declare_entry(
    "Functions",
    "x+y; x^2-y^3; x^2-y^3+0.03*(sin(3.14*x*12)+cos(3.14*y*21))",
    Patterns::Anything(),
    "Semicolon-separated functions for smoothing test");

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

  while (true)
    {
      const bool keep_going = solver.solve();

      smooth();
      solver.output_vtk();

      if (!keep_going)
        break;
    };
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Assert(dim == 2, ExcNotImplemented());

  Point<dim> p1, p2;
  p2[0]       = 1;
  p2[dim - 1] = 1;

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  triangulation.refine_global(3);
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();
  // no need to initialize the temperature field

  const std::string  vars    = FunctionParser<dim>::default_variable_names();
  const std::string  expr_f  = prm.get("Functions");
  const auto         f_split = Utilities::split_string_list(expr_f, ';');
  const unsigned int n_f     = f_split.size();

  function_values.reinit(n_f);
  functions.resize(n_f);

  for (unsigned int k = 0; k < n_f; ++k)
    {
      functions[k] = f_split[k];
      std::cout << "f_" << k << "(" << vars << ")=" << functions[k] << '\n';
    }

  calculate_f();

  for (unsigned int k = 0; k < n_f; ++k)
    solver.add_field("f_" + std::to_string(k), function_values.block(k));
}

template <int dim>
void
Problem<dim>::calculate_f()
{
  std::vector<Point<dim>> points;
  solver.get_support_points(points);

  const unsigned int n   = points.size();
  const unsigned int n_f = functions.size();

  for (unsigned int k = 0; k < n_f; ++k)
    {
      FunctionParser<dim> calc_f;
      calc_f.initialize(FunctionParser<dim>::default_variable_names(),
                        functions[k],
                        typename FunctionParser<dim>::ConstMap());

      Vector<double> &f = function_values.block(k);
      f.reinit(n);
      for (unsigned int i = 0; i < n; ++i)
        f[i] = calc_f.value(points[i]);
    }
}

template <int dim>
void
Problem<dim>::smooth()
{
  DoFFieldSmoother<dim> smoother;

  const unsigned int n_f = functions.size();

  for (unsigned int k = 0; k < n_f; ++k)
    {
      smoother.add_field("f_" + std::to_string(k), function_values.block(k));
    }

  smoother.attach_dof_handler(solver.get_dof_handler());
  smoother.calculate();

  for (unsigned int k = 0; k < n_f; ++k)
    {
      Vector<double> &f = function_values.block(k);

      f = smoother.get_field("f_" + std::to_string(k));
      solver.add_field("g_" + std::to_string(k), f);
    }
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init = false;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  for (unsigned int k = 1; k <= 3; ++k)
    {
      Problem<2> p(k, init);
      p.run();
    }

  return 0;
}
