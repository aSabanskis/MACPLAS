#include <deal.II/grid/grid_generator.h>

#include "../../include/temperature_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const std::vector<std::string> &arguments);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  unsigned int
  get_degree(const std::vector<std::string> &arguments) const;

  TemperatureSolver<dim> solver;

  std::string BC;

  bool steady;

  double vol_heat_source;
};

template <int dim>
Problem<dim>::Problem(const std::vector<std::string> &arguments)
  : solver(get_degree(arguments))
  , steady(false)
  , vol_heat_source(0)
{
  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (Utilities::match_at_string_start(arguments[i], "bc") ||
          Utilities::match_at_string_start(arguments[i], "BC"))
        BC = arguments[i].substr(2);

      if (arguments[i] == "steady")
        steady = true;

      if (arguments[i] == "vol_heat_source" && i + 1 < arguments.size())
        vol_heat_source = std::stod(arguments[i + 1]);
    }

  AssertThrow(!BC.empty(), ExcMessage("No boundary conditions provided"));

  std::cout << "BC = " << BC << '\n';
  std::cout << "vol_heat_source = " << vol_heat_source << '\n';
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

  if (!steady)
    return;

  // write results on x axis
  std::stringstream ss;
  ss << "result-" << dim << "d-order" << solver.get_degree() << "-x.dat";
  const std::string file_name = ss.str();
  std::cout << "Saving postprocessed results to '" << file_name << "'\n";

  const Vector<double> &temperature = solver.get_temperature();

  std::ofstream f(file_name);
  f << (dim == 1 ? "x" : dim == 2 ? "x y" : "x y z");
  f << " T[K]\n";
  f << std::setprecision(8);

  std::vector<Point<dim>> support_points;
  solver.get_support_points(support_points);

  for (unsigned int i = 0; i < support_points.size(); ++i)
    {
      // select points with y=z=0
      Point<dim> p = support_points[i];
      p[0]         = 0;
      if (p.norm_square() < 1e-8)
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

  // steady-state: set time step to zero
  if (steady)
    solver.get_time_step() = 0;

  solver.get_heat_source() = 0;
  solver.get_heat_source().add(vol_heat_source);

  unsigned int boundary_id = 0;

  if (BC == "1")
    {
      const double T0 = 1000;
      std::cout << "Setting T=" << T0 << " at boundary " << boundary_id << '\n';
      solver.set_bc1(boundary_id, T0);
      return;
    }

  if (BC == "2")
    {
      std::vector<Point<dim>> points;
      std::vector<bool>       boundary_dofs;

      const double q0 = 3e5;

      solver.get_boundary_points(boundary_id, points, boundary_dofs);
      Vector<double> q(points.size());
      for (unsigned int i = 0; i < q.size(); ++i)
        {
          if (boundary_dofs[i])
            q[i] = q0;
        }

      std::function<double(const double)> zero = [=](const double) {
        return 0.0;
      };

      std::cout << "Setting q=" << q0 << " at boundary " << boundary_id << '\n';
      solver.set_bc_rad_mixed(boundary_id, q, zero, zero);
      return;
    }

  if (BC == "3")
    {
      const double h = 2000;
      const double T = 1000;
      std::cout << "Setting h=" << h << ", T=" << T << " at boundary "
                << boundary_id << '\n';
      solver.set_bc_convective(boundary_id, h, T);
      return;
    }

  if (BC == "1b")
    {
      const double T1 = 500;
      const double T2 = 1000;
      std::cout << "Setting T=" << T1 << " at boundary 0\n";
      solver.set_bc1(0, T1);
      std::cout << "Setting T=" << T2 << " at boundary 1\n";
      solver.set_bc1(1, T2);
      return;
    }

  if (BC == "3b")
    {
      // Physical parameters from https://doi.org/10.1016/S0022-0248(03)01253-3
      const double T0 = 1687;

      Vector<double> &temperature = solver.get_temperature();
      temperature.add(T0);

      std::cout << "Setting T=" << T0 << " at boundary " << boundary_id << '\n';
      solver.set_bc1(boundary_id, T0);

      std::vector<Point<dim>> points;
      std::vector<bool>       boundary_dofs;
      boundary_id = 1;
      solver.get_boundary_points(boundary_id, points, boundary_dofs);
      const double   q0 = 1e4;
      Vector<double> q(points.size());
      for (unsigned int i = 0; i < q.size(); ++i)
        {
          if (boundary_dofs[i])
            q[i] = q0;
        }
      const double                        emissivity0 = 0.46;
      std::function<double(const double)> emissivity  = [=](const double T) {
        const double t = T / T0;
        return emissivity0 * (t < 0.593 ? 1.39 : 1.96 - 0.96 * t);
      };
      std::function<double(const double)> emissivity_deriv =
        [=](const double T) {
          const double t = T / T0;
          return emissivity0 * (t < 0.593 ? 0 : -0.96 / T0);
        };
      std::cout << "Setting e(T) at boundary " << boundary_id << '\n';
      std::cout << "Setting q=" << q0 << " at boundary " << boundary_id << '\n';
      solver.set_bc_rad_mixed(boundary_id, q, emissivity, emissivity_deriv);
    }
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

  int dimension = 1;

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

  if (dimension == 1)
    {
      Problem<1> p1(arguments);
      p1.run();
    }
  else if (dimension == 2)
    {
      Problem<2> p2(arguments);
      p2.run();
    }
  else if (dimension == 3)
    {
      Problem<3> p3(arguments);
      p3.run();
    }

  std::cout << "Finished\n";

  return 0;
}
