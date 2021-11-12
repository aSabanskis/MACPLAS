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

  Vector<double> f;

  FunctionParser<dim> calc_f;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : solver(order, use_default_prm)
  , calc_v(dim)
{
  prm.declare_entry("Function",
                    "tanh((1-x-y)*10)",
                    Patterns::Anything(),
                    "Function for advection test");

  prm.declare_entry("Velocity",
                    "0.02; 0.02",
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

  const std::string vars   = FunctionParser<dim>::default_variable_names();
  const std::string expr_f = prm.get("Function");
  const std::string expr_v = prm.get("Velocity");

  std::cout << "f(" << vars << ")=" << expr_f << '\n';
  std::cout << "v(" << vars << ")=" << expr_v << '\n';

  calc_f.initialize(vars, expr_f, typename FunctionParser<dim>::ConstMap());
  calc_v.initialize(vars, expr_v, typename FunctionParser<dim>::ConstMap());

  calculate_f_v();
  solver.add_field("f", f);
  solver.set_velocity(velocity);

  // add multiple functions to test implementation of BC1
  Vector<double> tmp(f.size());
  solver.add_field("another_function", tmp);
  tmp.add(1);
  solver.add_field("yet_another_function", tmp);

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

  f.reinit(n);
  velocity.resize(n);

  Vector<double> v(3);

  for (unsigned int i = 0; i < n; ++i)
    {
      f[i] = calc_f.value(points[i]);
      calc_v.vector_value(points[i], v);
      for (unsigned int k = 0; k < dim; ++k)
        velocity[i][k] = v[k];
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
