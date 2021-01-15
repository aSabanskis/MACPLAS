#include <deal.II/grid/grid_in.h>

#include <fstream>

#include "../../include/dislocation_solver.h"
#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

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
  apply_q_em(void);

  void
  calculate_temperature(void);

  TemperatureSolver<dim> temperature_solver;
  DislocationSolver<dim> dislocation_solver;

  // normalized Joulean heat flux density
  Vector<double> q0;

  constexpr static unsigned int boundary_id = 0;

  ParameterHandler prm;

  FunctionParser<1> electrical_conductivity;
};

template <int dim>
Problem<dim>::Problem(unsigned int order)
  : temperature_solver(order)
  , dislocation_solver(order)
{
  // Physical parameters from https://doi.org/10.1016/j.jcrysgro.2020.125842

  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature in K");

  prm.declare_entry("Max temperature change",
                    "0.1",
                    Patterns::Double(0),
                    "Initial temperature in K");

  prm.declare_entry("Inductor current",
                    "100",
                    Patterns::Double(0),
                    "Effective inductor current in A");

  prm.declare_entry("Reference electrical conductivity",
                    "5e4",
                    Patterns::Double(0),
                    "Reference electrical conductivity (qEM.vtu) in S/m");

  prm.declare_entry("Electrical conductivity",
                    "100*10^(4.247-2924.0/T)",
                    Patterns::Anything(),
                    "Electrical conductivity in S/m");

  prm.declare_entry("Emissivity",
                    "0.57",
                    Patterns::Double(0, 1),
                    "Emissivity (dimensionless)");

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

  electrical_conductivity.initialize("T",
                                     prm.get("Electrical conductivity"),
                                     typename FunctionParser<1>::ConstMap());
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize();

  calculate_temperature();

  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature_solver.get_temperature();
  dislocation_solver.solve(true);
  dislocation_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::calculate_temperature(void)
{
  double max_dT;

  do
    {
      Vector<double> temperature = temperature_solver.get_temperature();

      apply_q_em();
      temperature_solver.solve();

      temperature -= temperature_solver.get_temperature();
      max_dT = temperature.linfty_norm();

      std::cout << "max_dT=" << max_dT << " K\n";
    }
  while (max_dT > prm.get_double("Max temperature change"));

  temperature_solver.output_vtk();
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = temperature_solver.get_mesh();

  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh.msh");
  gi.read_msh(f);

  dislocation_solver.get_mesh().copy_triangulation(triangulation);
}

template <int dim>
void
Problem<dim>::initialize()
{
  temperature_solver.initialize(); // sets T=0

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Initial temperature"));

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  temperature_solver.get_boundary_points(boundary_id, points, boundary_dofs);

  SurfaceInterpolator3D surf;
  surf.read_vtu("qEM.vtu");
  surf.convert(SurfaceInterpolator3D::CellField,
               "QEM",
               SurfaceInterpolator3D::PointField,
               "q");
  surf.interpolate(
    SurfaceInterpolator3D::PointField, "q", points, boundary_dofs, q0);

  // normalize for future use
  q0 *= 1e-6 * std::sqrt(prm.get_double("Reference electrical conductivity"));
}

template <int dim>
void
Problem<dim>::apply_q_em(void)
{
  const Vector<double> &temperature = temperature_solver.get_temperature();

  Vector<double> q = q0;

  // apply the current and temperature-dependent electrical conductivity
  const double i2 = sqr(prm.get_double("Inductor current"));
  for (unsigned int i = 0; i < q.size(); ++i)
    {
      const double s = electrical_conductivity.value(Point<1>(temperature[i]));
      q[i] *= i2 / std::sqrt(s);
    }

  const double e = prm.get_double("Emissivity");

  std::function<double(double)> emissivity       = [=](double) { return e; };
  std::function<double(double)> emissivity_deriv = [=](double) { return 0.0; };

  temperature_solver.set_bc_rad_mixed(boundary_id,
                                      q,
                                      emissivity,
                                      emissivity_deriv);
}

int
main()
{
  Problem<3> p3d(2);
  p3d.run();

  return 0;
}
