#include <deal.II/grid/grid_generator.h>

#include "../../include/advection_solver.h"

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
  calculate_f_v();

  AdvectionSolver<dim> solver;

  std::vector<Tensor<1, dim>> velocity;

  FunctionParser<dim> calc_v;

  BlockVector<double> function_values;

  std::vector<std::string> functions;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
  , calc_v(dim)
{
  prm.declare_entry("Functions",
                    "0; 1; tanh(sin(x)+(y-0.5)^2)",
                    Patterns::Anything(),
                    "Semicolon-separated functions for advection test");

  prm.declare_entry("Velocity",
                    "-y*0.02; x*0.02",
                    Patterns::Anything(),
                    "Semicolon-separated velocity field in m/s");

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

      if (!keep_going)
        break;
    };

  solver.output_vtk();
  solver.output_boundary_values(0);
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Point<dim> p1;
  Point<dim> p2;

  for (unsigned int i = 0; i < dim; ++i)
    p2[i] = 1;

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  triangulation.refine_global(5);
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  const std::string vars = FunctionParser<dim>::default_variable_names();

  const std::string expr_v = prm.get("Velocity");
  std::cout << "v(" << vars << ")=" << expr_v << '\n';
  calc_v.initialize(vars, expr_v, typename FunctionParser<dim>::ConstMap());

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

  calculate_f_v();

  for (unsigned int k = 0; k < n_f; ++k)
    solver.add_field("f_" + std::to_string(k), function_values.block(k));

  solver.set_velocity(velocity);

  solver.set_bc1(0);

  solver.output_vtk();
  solver.output_boundary_values(0);
}

template <int dim>
void
Problem<dim>::calculate_f_v()
{
  std::vector<Point<dim>> points;
  solver.get_support_points(points);

  const unsigned int n = points.size();

  velocity.resize(n);

  Vector<double> v(3);

  for (unsigned int i = 0; i < n; ++i)
    {
      calc_v.vector_value(points[i], v);
      for (unsigned int k = 0; k < dim; ++k)
        velocity[i][k] = v[k];
    }

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

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool         init      = false;
  int          dimension = 2;
  unsigned int order     = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;

      if (arguments[i] == "1d" || arguments[i] == "1D")
        dimension = 1;
      if (arguments[i] == "2d" || arguments[i] == "2D")
        dimension = 2;
      if (arguments[i] == "3d" || arguments[i] == "3D")
        dimension = 3;

      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 1)
    {
      Problem<1> p1(order, init);
      if (!init)
        p1.run();
    }
  else if (dimension == 2)
    {
      Problem<2> p2(order, init);
      if (!init)
        p2.run();
    }
  else if (dimension == 3)
    {
      Problem<3> p3(order, init);
      if (!init)
        p3.run();
    }

  std::cout << "Finished\n";

  return 0;
}
