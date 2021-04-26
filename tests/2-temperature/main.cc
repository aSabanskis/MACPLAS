#include <deal.II/grid/grid_generator.h>

#include "../../include/temperature_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order = 2, const int bc = 0);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  TemperatureSolver<dim> solver;

  int BC;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const int bc)
  : solver(order)
  , BC(bc)
{}

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

  if (dim == 1)
    return;

  // write results on x axis
  std::stringstream ss;
  ss << "result-" << dim << "d-order"
     << solver.get_dof_handler().get_fe().degree << "-x.dat";
  const std::string file_name = ss.str();
  std::cout << "Saving postprocessed results to '" << file_name << "'";

  const Vector<double> &temperature = solver.get_temperature();

  std::ofstream f(file_name);
  f << (dim == 2 ? "x y" : "x y z");
  f << " T[K]\n";
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

      f << support_points[i] << " " << temperature[i] << '\n';
    }
}

template <int dim>
void
Problem<dim>::make_grid()
{
  const double x1 = dim == 2 ? 0.5 : 0;
  const double x2 = 1;

  Point<dim> p1;
  Point<dim> p2;

  p1[0] = p2[0] = x1;
  for (unsigned int i = 0; i < dim; ++i)
    p2[i] = x2;

  Triangulation<dim> &triangulation = solver.get_mesh();
  GridGenerator::hyper_rectangle(triangulation, p1, p2, true);

  triangulation.refine_global(4);

  const unsigned int n_probes = 11;
  for (unsigned int i = 0; i < n_probes; ++i)
    {
      Point<dim> p;
      p(0) = x1 + (x2 - x1) * i / (n_probes - 1);
      solver.add_probe(p);
    }
}

template <int dim>
void
Problem<dim>::initialize()
{
  solver.initialize(); // sets T=0

  unsigned int boundary_id = 0;

  // essential boundary conditions
  if (dim == 1 && BC == 1)
    {
      solver.set_bc1(boundary_id, 1000);
      return;
    }

  if (dim == 1 && BC == 2)
    {
      std::vector<Point<dim>> points;
      std::vector<bool>       boundary_dofs;

      solver.get_boundary_points(boundary_id, points, boundary_dofs);
      Vector<double> q(points.size());
      for (unsigned int i = 0; i < q.size(); ++i)
        {
          if (boundary_dofs[i])
            q[i] = 3e5;
        }

      std::function<double(const double)> zero = [=](const double) {
        return 0.0;
      };

      solver.set_bc_rad_mixed(boundary_id, q, zero, zero);
      return;
    }

  if (dim == 1 && BC == 3)
    {
      solver.set_bc_convective(boundary_id, 2000, 1000);
      return;
    }

  // steady-state
  solver.get_time_step() = 0;

  // natural boundary conditions
  if (dim == 2)
    {
      solver.set_bc1(0, 500);
      solver.set_bc1(1, 1000);
      return;
    }

  // BC == 3

  // Physical parameters from https://doi.org/10.1016/S0022-0248(03)01253-3
  const double T0 = 1687;

  Vector<double> &temperature = solver.get_temperature();
  temperature.add(T0);

  solver.set_bc1(boundary_id, T0);

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  boundary_id = 1;
  solver.get_boundary_points(boundary_id, points, boundary_dofs);
  Vector<double> q(points.size());
  for (unsigned int i = 0; i < q.size(); ++i)
    {
      if (boundary_dofs[i])
        q[i] = 1e4;
    }
  const double                        emissivity0 = 0.46;
  std::function<double(const double)> emissivity  = [=](const double T) {
    const double t = T / T0;
    return emissivity0 * (t < 0.593 ? 1.39 : 1.96 - 0.96 * t);
  };
  std::function<double(const double)> emissivity_deriv = [=](const double T) {
    const double t = T / T0;
    return emissivity0 * (t < 0.593 ? 0 : -0.96 / T0);
  };
  solver.set_bc_rad_mixed(boundary_id, q, emissivity, emissivity_deriv);
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  int order     = 2;
  int dimension = 1;
  int bc        = 0;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "1d" || arguments[i] == "1D")
        dimension = 1;
      if (arguments[i] == "2d" || arguments[i] == "2D")
        dimension = 2;
      if (arguments[i] == "3d" || arguments[i] == "3D")
        dimension = 3;

      if (arguments[i] == "bc1" || arguments[i] == "BC1")
        bc = 1;
      if (arguments[i] == "bc2" || arguments[i] == "BC2")
        bc = 2;
      if (arguments[i] == "bc3" || arguments[i] == "BC3")
        bc = 3;
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  std::cout << "Problem dimension=" << dimension << ", bc=" << bc << "\n";

  if (dimension == 1)
    {
      Problem<1> p(order, bc);
      p.run();
    }
  else if (dimension == 2)
    {
      Problem<2> p2(order);
      p2.run();
    }
  else if (dimension == 3)
    {
      Problem<3> p3(order);
      p3.run();
    }

  return 0;
}
