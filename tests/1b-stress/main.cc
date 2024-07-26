#include <deal.II/grid/grid_generator.h>

#include "../../include/stress_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  explicit Problem(const std::vector<std::string> &arguments);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  unsigned int
  get_degree(const std::vector<std::string> &arguments) const;

  StressSolver<dim> solver;

  std::string BC;

  double Lx, Ly, Lz;

  int refine_global, refine_x;

  unsigned int   boundary;
  unsigned int   component;
  double         displacement;
  Tensor<1, dim> load;
};

template <int dim>
Problem<dim>::Problem(const std::vector<std::string> &arguments)
  : solver(get_degree(arguments))
  , Lx(0.1)
  , Ly(0.015)
  , Lz(0.003)
  , refine_global(1)
  , refine_x(5)
  , boundary(1)
  , component(0)
  , displacement(0)
{
  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (Utilities::match_at_string_start(arguments[i], "bc") ||
          Utilities::match_at_string_start(arguments[i], "BC"))
        BC = arguments[i].substr(2);

      if (arguments[i] == "Lx" && i + 1 < arguments.size())
        Lx = std::stod(arguments[i + 1]);

      if (arguments[i] == "Ly" && i + 1 < arguments.size())
        Ly = std::stod(arguments[i + 1]);

      if (arguments[i] == "Lz" && i + 1 < arguments.size())
        Lz = std::stod(arguments[i + 1]);

      if (arguments[i] == "refine_global" && i + 1 < arguments.size())
        refine_global = std::stoi(arguments[i + 1]);

      if (arguments[i] == "refine_x" && i + 1 < arguments.size())
        refine_x = std::stoi(arguments[i + 1]);

      if (arguments[i] == "boundary" && i + 1 < arguments.size())
        boundary = std::stoi(arguments[i + 1]);

      if (arguments[i] == "component" && i + 1 < arguments.size())
        component = std::stoi(arguments[i + 1]);

      if (arguments[i] == "displacement" && i + 1 < arguments.size())
        displacement = std::stod(arguments[i + 1]);

      if (arguments[i] == "load" && i + dim < arguments.size())
        for (unsigned int k = 0; k < dim; ++k)
          load[k] = std::stod(arguments[i + 1 + k]);
    }

  AssertThrow(!BC.empty(), ExcMessage("No boundary conditions provided"));

  std::cout << "BC = " << BC << '\n';
  std::cout << "refine_global = " << refine_global << '\n';
  std::cout << "refine_x = " << refine_x << '\n';
  std::cout << "boundary = " << boundary << '\n';
  std::cout << "component = " << component << '\n';
  std::cout << "displacement = " << displacement << '\n';
  std::cout << "load = " << load << '\n';
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  solver.solve();
  solver.output_vtk();

  // write results on x axis
  std::stringstream ss;
  ss << "result-" << dim << "d-order"
     << solver.get_dof_handler().get_fe().degree << "-BC" << BC << boundary
     << "-";
  if (BC == "d")
    ss << displacement;
  else if (BC == "f")
    {
      ss << load[0];
      for (unsigned int i = 1; i < dim; ++i)
        ss << "_" << load[i];
    }
  ss << "-x.dat";
  const std::string file_name = ss.str();
  std::cout << "Saving postprocessed results to '" << file_name << "'\n";

  const BlockVector<double> &d = solver.get_displacement();

  std::ofstream f(file_name);
  f << (dim == 2 ? "x y" : "x y z");
  for (unsigned int k = 0; k < d.n_blocks(); ++k)
    f << " d_" << k << "[m]";
  f << '\n';
  f << std::setprecision(8);

  std::vector<Point<dim>> support_points;
  solver.get_support_points(support_points);

  for (unsigned int i = 0; i < support_points.size(); ++i)
    {
      // select points with y=z=0
      Point<dim> p = support_points[i];
      p[0]         = 0;
      if (p.norm_square() > 1e-8)
        continue;

      f << support_points[i];
      for (unsigned int k = 0; k < d.n_blocks(); ++k)
        f << " " << d.block(k)[i];
      f << '\n';
    }
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Point<dim> p1; // 0 0 0
  Point<dim> p2;

  Point<3> pMax(Lx, Ly, Lz);

  for (unsigned int i = 0; i < dim; ++i)
    p2[i] = pMax[i];

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::hyper_rectangle(triangulation, p1, p2, true);

  triangulation.refine_global(refine_global);

  for (int i = 0; i < refine_x; ++i)
    {
      typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
      for (; cell != endc; ++cell)
        cell->set_refine_flag(RefinementCase<dim>::cut_x);

      triangulation.execute_coarsening_and_refinement();
    }
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize();

  // set boundary conditions at x=0 and x=L
  for (unsigned int i = 0; i < dim; ++i)
    solver.set_bc1(0, i, 0);

  if (BC == "d")
    solver.set_bc1(boundary, component, displacement);
  else if (BC == "f")
    solver.set_bc_load(boundary, load);
  else
    AssertThrow(false,
                ExcMessage("initialize: BC '" + BC + "' not supported."));
}

template <int dim>
unsigned int
Problem<dim>::get_degree(const std::vector<std::string> &arguments) const
{
  unsigned int order = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    if (arguments[i] == "order" && i + 1 < arguments.size())
      order = std::stoi(arguments[i + 1]);

  return order;
}


int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  int dimension = 3;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "1d" || arguments[i] == "1D")
        dimension = 1;
      if (arguments[i] == "2d" || arguments[i] == "2D")
        dimension = 2;
      if (arguments[i] == "3d" || arguments[i] == "3D")
        dimension = 3;
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 2)
    {
      Problem<2> p2(arguments);
      p2.run();
    }
  else if (dimension == 3)
    {
      Problem<3> p3(arguments);
      p3.run();
    }

  return 0;
}
