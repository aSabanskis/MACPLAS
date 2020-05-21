#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "../../include/stress_solver.h"

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
  output_results() const;

  double r_a, r_b;
  double T_a, T_b;

  Point<dim>                   center;
  const SphericalManifold<dim> manifold;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe_temp;
  DoFHandler<dim>    dh_temp;

  Vector<double> temperature;

  StressSolver<dim> solver;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : r_a(0.5)
  , r_b(1.0)
  , T_a(500)
  , T_b{1000}
  , center(Point<dim>())
  , manifold(SphericalManifold<dim>(center))
  , fe_temp(order)
  , dh_temp(triangulation)
  , solver(order)
{}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();
  output_results();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Assert(r_a > 0, ExcMessage("r_a must be positive"));
  Assert(r_b > r_a, ExcMessage("r_b must be larger than r_b"));
  GridGenerator::hyper_shell(triangulation, center, r_a, r_b, 0, true);

  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold(0, manifold);
  triangulation.set_manifold(1, manifold);

  triangulation.refine_global(3);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << "\n";
}

template <int dim>
void
Problem<dim>::initialize()
{
  dh_temp.distribute_dofs(fe_temp);
  const unsigned int n_dofs = dh_temp.n_dofs();
  std::cout << "Number of degrees of freedom: " << n_dofs << "\n";

  temperature.reinit(n_dofs);

  std::vector<Point<dim>> support_points(n_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                       dh_temp,
                                       support_points);

  // Set the temperature using analytical solution in spherical coordinates.
  const double c1 = r_a * (T_a - T_b) / (1 - r_a / r_b);
  const double c2 = (T_b - T_a * r_a / r_b) / (1 - r_a / r_b);
  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      temperature[i] = c1 / support_points[i].norm() + c2;
    }
}

template <int dim>
void
Problem<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh_temp);
  data_out.add_data_vector(temperature, "T");

  data_out.build_patches();

  const std::string file_name = "result-" + std::to_string(dim) + "d.vtk";
  std::cout << "Saving to " << file_name << "\n";

  std::ofstream output(file_name);
  data_out.write_vtk(output);
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}
