#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <fstream>

#include "../../include/temperature_solver.h"

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
  initialize_temperature();

  TemperatureSolver<dim> temperature_solver;

  Triangulation<dim> triangulation;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : temperature_solver(order, use_default_prm)
{
  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize_temperature();

  temperature_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh-" + std::to_string(dim) + "d.msh");
  gi.read_msh(f);

  temperature_solver.get_mesh().copy_triangulation(triangulation);

  // print info
  std::cout << "Number of cells: " << triangulation.n_cells() << "\n"
            << "Number of points: " << triangulation.n_vertices() << "\n";

  const auto bounds = triangulation.get_boundary_ids();
  for (const auto &b : bounds)
    std::cout << "Boundary no. " << (int)b << "\n";
}

template <int dim>
void
Problem<dim>::initialize_temperature()
{
  temperature_solver.initialize(); // sets T=0

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Initial temperature"));

  temperature_solver.output_mesh();
  temperature_solver.output_parameter_table();
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init      = false;
  int  order     = 2;
  int  dimension = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 2)
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
