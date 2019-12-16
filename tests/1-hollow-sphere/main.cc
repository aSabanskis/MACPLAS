#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

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

  Point<dim>                   center;
  const SphericalManifold<dim> manifold;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  Vector<double> temperature;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : center(Point<dim>())
  , manifold(SphericalManifold<dim>(center))
  , fe(order)
  , dof_handler(triangulation)
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
  GridGenerator::hyper_shell(triangulation, center, 0.5, 1.0, 0, true);

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
  dof_handler.distribute_dofs(fe);
  const unsigned int n_dofs = dof_handler.n_dofs();
  std::cout << "Number of degrees of freedom: " << n_dofs << "\n";

  temperature.reinit(n_dofs);

  std::vector<Point<dim>> support_points(n_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                       dof_handler,
                                       support_points);

  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      temperature[i] = support_points[i].norm();
    }
}

template <int dim>
void
Problem<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
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
  Problem<2> p2d;
  p2d.run();

  Problem<3> p3d;
  p3d.run();

  return 0;
}
