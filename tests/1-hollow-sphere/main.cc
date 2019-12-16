#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

using namespace dealii;

class Problem
{
public:
  Problem();

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  void
  output_results() const;

  Point<2>                   center;
  const SphericalManifold<2> manifold;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  Vector<double> temperature;
};

Problem::Problem()
  : center(Point<2>())
  , manifold(SphericalManifold<2>(center))
  , fe(2)
  , dof_handler(triangulation)
{}

void
Problem::run()
{
  make_grid();
  initialize();
  output_results();
}

void
Problem::make_grid()
{
  GridGenerator::hyper_shell(triangulation, center, 0.5, 1.0, 0, true);

  triangulation.set_manifold(0, manifold);
  triangulation.set_manifold(1, manifold);

  triangulation.refine_global(3);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << "\n";
}

void
Problem::initialize()
{
  dof_handler.distribute_dofs(fe);
  const unsigned int n_dofs = dof_handler.n_dofs();
  std::cout << "Number of degrees of freedom: " << n_dofs << "\n";

  temperature.reinit(n_dofs);

  std::vector<Point<2>> support_points(n_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<2>(),
                                       dof_handler,
                                       support_points);

  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      temperature[i] = support_points[i].norm();
    }
}

void
Problem::output_results() const
{
  DataOut<2> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(temperature, "T");

  data_out.build_patches();
  std::ofstream output("result.vtk");
  data_out.write_vtk(output);
}

int
main()
{
  Problem p;
  p.run();

  return 0;
}
