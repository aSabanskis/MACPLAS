#include <deal.II/grid/grid_generator.h>

#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  explicit Problem(const unsigned int order = 2);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  void
  calculate_gradient();

  TemperatureSolver<dim> solver;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order)
  : solver(order)
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();
  calculate_gradient();
  solver.output_vtk();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Assert(dim == 2, ExcNotImplemented());

  Point<dim> p1, p2;
  p2[0]       = 2;
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
}

template <int dim>
void
Problem<dim>::calculate_gradient()
{
  std::vector<Point<dim>> points;
  solver.get_support_points(points);

  const unsigned int n    = points.size();
  const auto         dims = coordinate_names(dim);

  Vector<double> f(n);

  std::vector<Vector<double>> grad_f(dim, Vector<double>(n));

  for (unsigned int i = 0; i < n; ++i)
    {
      f[i] = -sqr(points[i][0]) -
             std::cos(2 * points[i][0]) * std::sqrt(points[i][dim - 1] + 1.5);
      // calculate the reference gradient (analytical)
      grad_f[0][i] = -2 * points[i][0] + 2 * std::sin(2 * points[i][0]) *
                                           std::sqrt(points[i][dim - 1] + 1.5);
      grad_f[dim - 1][i] =
        -std::cos(2 * points[i][0]) / (2 * std::sqrt(points[i][dim - 1] + 1.5));
    }

  solver.add_field("f", f);
  for (unsigned int k = 0; k < dim; ++k)
    solver.add_field("df_d" + dims[k] + "_ref", grad_f[k]);

  DoFGradientEvaluation<dim> grad_eval;

  grad_eval.add_field("f", f);
  grad_eval.attach_dof_handler(solver.get_dof_handler());
  grad_eval.calculate();
  const auto &grad_calc = grad_eval.get_gradient("f");

  for (unsigned int k = 0; k < dim; ++k)
    {
      for (unsigned int i = 0; i < n; ++i)
        grad_f[k][i] = grad_calc[i][k];

      solver.add_field("df_d" + dims[k] + "_calc", grad_f[k]);
    }
}

int
main()
{
  deallog.attach(std::cout);
  deallog.depth_console(2);

  for (unsigned int k = 1; k <= 3; ++k)
    {
      Problem<2> p(k);
      p.run();
    }

  return 0;
}
