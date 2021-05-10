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
  Problem(const unsigned int order = 2, const bool use_default_prm = false);

  void
  run();

private:
  void
  make_grid();

  void
  initialize();

  void
  apply_q_em();

  void
  interpolate_q_em(const double z);

  void
  solve_steady_temperature();

  void
  solve_dislocation();

  void
  solve_temperature_dislocation();

  void
  solve_temperature();

  void
  output_results(const bool data     = true,
                 const bool vtk      = true,
                 const bool boundary = dim == 2) const;

  void
  update_T_max();

  // false if only the temperature field is calculated
  bool
  with_dislocation() const;

  TemperatureSolver<dim> temperature_solver;
  DislocationSolver<dim> dislocation_solver;

  // external Joulean heat flux density data
  SurfaceInterpolator2D q2d;
  SurfaceInterpolator3D q3d;

  // normalized Joulean heat flux density
  Vector<double> q0;

  // max temperature during the whole simulation
  Vector<double> T_max;

  constexpr static unsigned int boundary_id = 0;

  FunctionParser<1> inductor_position;
  FunctionParser<1> inductor_current;

  ParameterHandler prm;

  FunctionParser<1> electrical_conductivity;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : temperature_solver(order, use_default_prm)
  , dislocation_solver(order, use_default_prm)
{
  // Physical parameters from https://doi.org/10.1016/j.jcrysgro.2020.125842

  prm.declare_entry("Initial temperature",
                    "1000",
                    Patterns::Double(0),
                    "Initial temperature T_0 in K");

  prm.declare_entry("Max temperature change",
                    "0.1",
                    Patterns::Double(0),
                    "Maximum temperature change in K");

  prm.declare_entry("Inductor position",
                    "0",
                    Patterns::Anything(),
                    "Vertical inductor shift in m (time function)");

  prm.declare_entry("Inductor current",
                    "100",
                    Patterns::Anything(),
                    "Effective inductor current I in A (time function)");

  prm.declare_entry(
    "Reference electrical conductivity",
    "5e4",
    Patterns::Double(0),
    "Reference electrical conductivity sigma_ref (qEM data file) in S/m");

  prm.declare_entry(
    "Electrical conductivity",
    "100*10^(4.247-2924.0/T)",
    Patterns::Anything(),
    "Electrical conductivity sigma in S/m (temperature function)");

  prm.declare_entry("Emissivity",
                    "0.57",
                    Patterns::Double(0, 1),
                    "Emissivity epsilon (dimensionless)");

  prm.declare_entry("Load saved results",
                    "false",
                    Patterns::Bool(),
                    "Skip calculation of temperature and stress fields");

  prm.declare_entry("Temperature only",
                    "false",
                    Patterns::Bool(),
                    "Calculate just the temperature field");

  prm.declare_entry(
    "Output frequency",
    "0",
    Patterns::Integer(0),
    "Number of time steps between result output (0 - disabled)");

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

  inductor_position.initialize("t",
                               prm.get("Inductor position"),
                               typename FunctionParser<1>::ConstMap());

  inductor_current.initialize("t",
                              prm.get("Inductor current"),
                              typename FunctionParser<1>::ConstMap());

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

  if (!with_dislocation())
    {
      if (temperature_solver.get_time_step() == 0)
        {
          solve_steady_temperature();
        }
      else
        {
          solve_temperature();
        }
    }
  else if (temperature_solver.get_time_step() == 0)
    {
      solve_steady_temperature();
      solve_dislocation();
    }
  else
    {
      solve_temperature_dislocation();
    }
}

template <int dim>
void
Problem<dim>::solve_steady_temperature()
{
  if (prm.get_bool("Load saved results"))
    {
      temperature_solver.load_data();
      return;
    }

  double max_dT;
  do
    {
      Vector<double> temperature = temperature_solver.get_temperature();

      apply_q_em();
      temperature_solver.solve();

      temperature -= temperature_solver.get_temperature();
      max_dT = temperature.linfty_norm();

      std::cout << "max_dT=" << max_dT << " K\n";
  } while (max_dT > prm.get_double("Max temperature change"));

  update_T_max();

  output_results();
}

template <int dim>
void
Problem<dim>::solve_dislocation()
{
  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature_solver.get_temperature();

  update_T_max();

  if (prm.get_bool("Load saved results"))
    {
      dislocation_solver.load_data();
    }
  else
    {
      output_results();
    }

  dislocation_solver.solve(true);

  while (true)
    {
      const bool keep_going = dislocation_solver.solve();

      if (!keep_going)
        break;
    };

  output_results();
}

template <int dim>
void
Problem<dim>::solve_temperature_dislocation()
{
  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature_solver.get_temperature();

  update_T_max();

  dislocation_solver.solve(true);

  const int n_output = prm.get_integer("Output frequency");

  for (unsigned int i = 1;; ++i)
    {
      temperature_solver.get_time_step() = dislocation_solver.get_time_step();

      apply_q_em();
      const bool keep_going_temp = temperature_solver.solve();

      dislocation_solver.get_temperature() =
        temperature_solver.get_temperature();

      update_T_max();

      const bool keep_going_disl = dislocation_solver.solve();

      if (!keep_going_temp || !keep_going_disl)
        break;

      if (n_output > 0 && i % n_output == 0)
        {
          output_results(false);
        }
    };

  output_results(false);
}

template <int dim>
void
Problem<dim>::solve_temperature()
{
  const int n_output = prm.get_integer("Output frequency");

  update_T_max();

  for (unsigned int i = 1;; ++i)
    {
      apply_q_em();
      const bool keep_going_temp = temperature_solver.solve();

      update_T_max();

      if (!keep_going_temp)
        break;

      if (n_output > 0 && i % n_output == 0)
        {
          output_results(false);
        }
    };

  output_results(false);
}

template <int dim>
void
Problem<dim>::output_results(const bool data,
                             const bool vtk,
                             const bool boundary) const
{
  if (data)
    {
      temperature_solver.output_data();
      if (with_dislocation())
        dislocation_solver.output_data();
    }

  if (vtk)
    {
      temperature_solver.output_vtk();
      if (with_dislocation())
        dislocation_solver.output_vtk();
    }

  // Exports values at the crystal surface and on axis in 2D.
  // Export is disabled by default in 3D.
  if (boundary)
    for (unsigned int id = 0; id < 2; ++id)
      {
        temperature_solver.output_boundary_values(id);
        if (with_dislocation())
          dislocation_solver.output_boundary_values(id);
      }
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = temperature_solver.get_mesh();

  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh-" + std::to_string(dim) + "d.msh");
  gi.read_msh(f);

  dislocation_solver.get_mesh().copy_triangulation(triangulation);
}

template <int dim>
void
Problem<dim>::initialize()
{
  temperature_solver.initialize(); // sets T=0

  temperature_solver.output_mesh();

  Point<dim> p;
  for (int i = 0; i <= 2; ++i)
    {
      p[dim - 1] = 0.067 + 0.2 * i;
      temperature_solver.add_probe(p);
      dislocation_solver.add_probe(p);
    }

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Initial temperature"));

  interpolate_q_em(inductor_position.value(Point<1>(0)));

  if (prm.get_bool("Load saved results"))
    {
      temperature_solver.load_data();
      return;
    }
}

template <int dim>
void
Problem<dim>::apply_q_em()
{
  const double t =
    temperature_solver.get_time() + temperature_solver.get_time_step();

  const Vector<double> &temperature = temperature_solver.get_temperature();

  const double z = inductor_position.value(Point<1>(t));
  temperature_solver.add_output("z[m]", z);

  interpolate_q_em(z);

  Vector<double> q = q0;

  const double I = inductor_current.value(Point<1>(t));
  temperature_solver.add_output("I[A]", I);

  // apply the current and temperature-dependent electrical conductivity
  const double I2 = sqr(I);
  for (unsigned int i = 0; i < q.size(); ++i)
    {
      const double s = electrical_conductivity.value(Point<1>(temperature[i]));
      q[i] *= I2 / std::sqrt(s);
    }

  const double e = prm.get_double("Emissivity");

  std::function<double(double)> emissivity       = [=](double) { return e; };
  std::function<double(double)> emissivity_deriv = [=](double) { return 0.0; };

  temperature_solver.set_bc_rad_mixed(boundary_id,
                                      q,
                                      emissivity,
                                      emissivity_deriv);
}

template <>
void
Problem<3>::interpolate_q_em(const double z)
{
  std::vector<Point<3>> points;
  std::vector<bool>     boundary_dofs;
  temperature_solver.get_boundary_points(boundary_id, points, boundary_dofs);

  for (auto &p : points)
    p[2] -= z;

  q3d.read_vtu("qEM-3d.vtu");
  q3d.convert(SurfaceInterpolator3D::CellField,
              "QEM",
              SurfaceInterpolator3D::PointField,
              "q");
  q3d.interpolate(
    SurfaceInterpolator3D::PointField, "q", points, boundary_dofs, q0);

  // normalize for future use
  q0 *= 1e-6 * std::sqrt(prm.get_double("Reference electrical conductivity"));
}

template <>
void
Problem<2>::interpolate_q_em(const double z)
{
  std::vector<Point<2>> points;
  std::vector<bool>     boundary_dofs;
  temperature_solver.get_boundary_points(boundary_id, points, boundary_dofs);

  for (auto &p : points)
    p[1] -= z;

  q2d.read_txt("qEM-2d.txt");
  q2d.interpolate("QEM", points, boundary_dofs, q0);

  // normalize for future use
  q0 *= 1e-6 * std::sqrt(prm.get_double("Reference electrical conductivity"));
}

template <int dim>
void
Problem<dim>::update_T_max()
{
  const Vector<double> &T = temperature_solver.get_temperature();

  if (T_max.size() == 0)
    {
      std::cout << "Initializing T_max\n";
      T_max = T;
    }

  AssertDimension(T_max.size(), T.size());

  for (unsigned int k = 0; k < T_max.size(); ++k)
    T_max[k] = std::max(T_max[k], T[k]);

  temperature_solver.add_field("T_max", T_max);
}

template <int dim>
bool
Problem<dim>::with_dislocation() const
{
  return !prm.get_bool("Temperature only");
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init      = false;
  int  order     = 2;
  int  dimension = 3;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
      if (arguments[i] == "2d" || arguments[i] == "2D")
        dimension = 2;
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 3)
    {
      Problem<3> p3d(order, init);
      if (!init)
        p3d.run();
    }
  else if (dimension == 2)
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
