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
  initialize_temperature();

  void
  initialize_dislocation();

  void
  apply_T_BC();

  void
  interpolate_q_em(const double z);

  /** Reduces the current time step to minimum value at user-defined instants
   * (e.g., when the inductor position starts to change)
   */
  void
  update_min_time_step();

  /** Checks whether the results need to be outputted at the end of the current
   * time step, and adjusts it if necessary */
  bool
  update_output_time_step(const unsigned int time_step_index = 0);

  void
  set_time_step(const double dt);

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
  postprocess();

  void
  update_T_max();

  void
  update_tau_eff_max();

  void
  measure_T();

  void
  approximate_LF_EM();

  // true if first-type temperature BC is applied at crystal surface
  bool
  T_BC1_applied() const;

  // false if only the temperature field is calculated
  bool
  with_dislocation() const;

  Timer timer;

  TemperatureSolver<dim> temperature_solver;
  DislocationSolver<dim> dislocation_solver;

  // external Joulean heat flux density data
  SurfaceInterpolator2D q2d;
  SurfaceInterpolator3D q3d;

  // external temperature BC
  SurfaceInterpolator2D T2d;

  // normalized Joulean heat flux density
  Vector<double> q0;

  // max temperature during the whole simulation
  Vector<double> T_max;

  // temperature at the previous time step
  Vector<double> T_prev;

  // min and max temperature time derivative during the whole simulation
  Vector<double> dot_T_min, dot_T_max;

  // max effective stress during the whole simulation
  Vector<double> tau_eff_max;

  std::vector<Point<dim>> inductor_probes;

  std::ofstream inductor_probe_file;

  constexpr static unsigned int boundary_id_surf = 0;
  constexpr static unsigned int boundary_id_axis = 1;

  std::unique_ptr<Function<1>> inductor_position;
  std::unique_ptr<Function<1>> inductor_current;

  double previous_inductor_position;

  double previous_time_step;
  double next_output_time;

  std::vector<double> dt_min_times;

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

  prm.declare_entry(
    "Reference inductor position",
    "0.26685",
    Patterns::Double(),
    "Reference vertical inductor position (qEM data file) in m");

  prm.declare_entry(
    "Inductor position",
    "0",
    Patterns::Anything(),
    "Vertical inductor shift in m (time function or data file name)");

  prm.declare_entry(
    "Inductor current",
    "100",
    Patterns::Anything(),
    "Effective inductor current I in A (time function or data file name)");

  prm.declare_entry("Inductor frequency",
                    "2.715e6",
                    Patterns::Double(0),
                    "Inductor frequency f in Hz");

  prm.declare_entry("QEM scale",
                    "1e-6",
                    Patterns::Double(),
                    "Scale for induced heat sources");

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
                    Patterns::Anything(),
                    "Emissivity epsilon (dimensionless)");

  prm.declare_entry("Ambient temperature",
                    "0",
                    Patterns::Double(0),
                    "Ambient temperature T_amb in K");

  prm.declare_entry("Heat transfer coefficient",
                    "0",
                    Patterns::Double(0),
                    "Heat transfer coefficient h in W/m^2/K");

  prm.declare_entry("Reference temperature",
                    "1000",
                    Patterns::Double(0),
                    "Reference temperature T_ref in K");


  prm.declare_entry("Load saved results",
                    "false",
                    Patterns::Bool(),
                    "Skip calculation of temperature and stress fields");

  prm.declare_entry("Temperature only",
                    "false",
                    Patterns::Bool(),
                    "Calculate just the temperature field");

  prm.declare_entry("Start from steady temperature",
                    "true",
                    Patterns::Bool(),
                    "Calculate the steady-state temperature field at t=0");

  prm.declare_entry(
    "Steady Newton step length",
    "0.6",
    Patterns::Double(0, 1),
    "Newton step length for steady-state temperature calculation");

  prm.declare_entry(
    "Outer temperature iterations",
    "1",
    Patterns::Integer(1),
    "Number of outer temperature iterations (needed for EM BC update)");

  prm.declare_entry(
    "Approximate skin effect",
    "false",
    Patterns::Bool(),
    "Approximate skin effect for temperature (for testing only)");

  prm.declare_entry("Use LF EM field",
                    "false",
                    Patterns::Bool(),
                    "Approximate LF EM field from HF EM data");

  prm.declare_entry("Max time",
                    "0",
                    Patterns::Double(0),
                    "Maximum time in seconds (0 - use solver parameters)");

  prm.declare_entry(
    "Output frequency",
    "0",
    Patterns::Integer(0),
    "Number of time steps between result output (0 - disabled)");

  prm.declare_entry("Output time step",
                    "0",
                    Patterns::Double(0),
                    "Time interval in s between result output (0 - disabled). "
                    "The time step is adjusted to match the required time. "
                    "Overrides 'Output frequency'.");

  prm.declare_entry(
    "Time step reduction instants",
    "",
    Patterns::List(Patterns::Double()),
    "Comma-separated times in s at which minimum time step should be applied");

  prm.declare_entry("Probe coordinates x",
                    "0, 0, 0",
                    Patterns::List(Patterns::Double(), 0),
                    "Comma-separated radial coordinates");

  prm.declare_entry("Probe coordinates z",
                    "0.067, 0.267, 0.467",
                    Patterns::List(Patterns::Double(), 0),
                    "Comma-separated vertical coordinates");

  prm.declare_entry("Custom probes x relative to inductor",
                    "0",
                    Patterns::List(Patterns::Double(), 0),
                    "Comma-separated radial coordinates");

  prm.declare_entry("Custom probes z relative to inductor",
                    "0",
                    Patterns::List(Patterns::Double(), 0),
                    "Comma-separated vertical coordinates");

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

  const double t_max = prm.get_double("Max time");
  if (t_max > 0)
    {
      temperature_solver.get_parameters().set("Max time", t_max);
      dislocation_solver.get_parameters().set("Max time", t_max);
    }

  // print all problem-specific and solver parameters
  std::cout << "# ---------------------\n"
            << "# Problem\n";
  prm.print_parameters(std::cout, ParameterHandler::Text);

  std::cout << "# ---------------------\n"
            << "# " << temperature_solver.solver_name() << '\n';
  temperature_solver.get_parameters().print_parameters(std::cout,
                                                       ParameterHandler::Text);

  if (with_dislocation())
    {
      std::cout << "# ---------------------\n"
                << "# " << dislocation_solver.solver_name() << '\n';
      dislocation_solver.get_parameters().print_parameters(
        std::cout, ParameterHandler::Text);

      std::cout << "# ---------------------\n"
                << "# " << dislocation_solver.get_stress_solver().solver_name()
                << '\n';
      dislocation_solver.get_stress_solver().get_parameters().print_parameters(
        std::cout, ParameterHandler::Text);
    }

  initialize_function(inductor_position, prm.get("Inductor position"));

  initialize_function(inductor_current, prm.get("Inductor current"));

  electrical_conductivity.initialize("T",
                                     prm.get("Electrical conductivity"),
                                     typename FunctionParser<1>::ConstMap());

  const std::vector<double> X =
    split_string(prm.get("Custom probes x relative to inductor"));
  const std::vector<double> Z =
    split_string(prm.get("Custom probes z relative to inductor"));
  AssertDimension(X.size(), Z.size());

  inductor_probes.resize(X.size());

  for (unsigned int i = 0; i < X.size(); ++i)
    {
      inductor_probes[i][0]       = X[i];
      inductor_probes[i][dim - 1] = Z[i];
    }

  next_output_time = prm.get_double("Output time step");
}

template <int dim>
void
Problem<dim>::run()
{
  make_grid();
  initialize_temperature();

  if (prm.get_bool("Start from steady temperature") ||
      temperature_solver.get_time_step() == 0)
    {
      solve_steady_temperature();
    }

  if (!with_dislocation())
    {
      if (temperature_solver.get_time_step() > 0)
        {
          solve_temperature();
        }
    }
  else if (temperature_solver.get_time_step() == 0)
    {
      solve_dislocation();
    }
  else
    {
      solve_temperature_dislocation();
    }

  std::cout << "Finished in " << timer.wall_time() << " s\n";
}

template <int dim>
void
Problem<dim>::update_min_time_step()
{
  const double dt_min = dislocation_solver.get_time_step_min();

  if (!with_dislocation() || dt_min == 0)
    return;

  const double t  = dislocation_solver.get_time();
  const double dt = dislocation_solver.get_time_step();

  bool found_time = false;

  double dt_new;

  for (const auto &t_new : dt_min_times)
    {
      if (t_new < t)
        continue;

      if (!found_time)
        {
          dt_new     = t_new - t;
          found_time = true;
        }
      else
        {
          dt_new = std::min(dt_new, t_new - t);
        }
    }

  if (found_time && dt_new < dt)
    {
      const double t_new = t + dt_new;

      dt_new = std::max(dt_new, dt_min);

      set_time_step(dt_new);

      std::cout << "dt changed from " << dt << " to " << dt_new
                << " s at user-specified time instant" << t_new << " s\n";
    }
}

template <int dim>
bool
Problem<dim>::update_output_time_step(const unsigned int time_step_index)
{
  const double t  = temperature_solver.get_time();
  const double dt = temperature_solver.get_time_step();

  const double dt_output = prm.get_double("Output time step");
  const int    n_output  = prm.get_integer("Output frequency");

  previous_time_step = dt;

  // output time step is not specified, check the index
  if (dt_output <= 0)
    {
      return time_step_index > 0 && n_output > 0 &&
             time_step_index % n_output == 0;
    }

  // output time too far away, nothing to do
  if (dt_output > 0 && t + (1 + 1e-4) * dt < next_output_time)
    {
      return false;
    }

  // output time is specified and the time step has to be adjusted
  set_time_step(next_output_time - t);

  std::cout << "dt changed from " << dt << " to "
            << temperature_solver.get_time_step()
            << " s for result output at t=" << next_output_time << " s\n";

  next_output_time += dt_output;

  return true;
}

template <int dim>
void
Problem<dim>::set_time_step(const double dt)
{
  temperature_solver.get_time_step() = dt;
  dislocation_solver.get_time_step() = dt;
}

template <int dim>
void
Problem<dim>::solve_steady_temperature()
{
  if (prm.get_bool("Load saved results"))
    {
      temperature_solver.load_data();
      postprocess();
      output_results();
      return;
    }

  std::cout << "Calculating steady-state temperature field\n";

  const double dt0 = temperature_solver.get_time_step();

  ParameterHandler &prm_T = temperature_solver.get_parameters();
  // saving as string to avoid ambiguous overloads
  const std::string n0 = prm_T.get("Max Newton iterations");
  const std::string a0 = prm_T.get("Newton step length");

  temperature_solver.get_time_step() = 0;
  // just one iteration to update external Joulean heat flux BC
  prm_T.set("Max Newton iterations", "1");
  // reduce step length to avoid divergence
  prm_T.set("Newton step length", prm.get("Steady Newton step length"));

  double max_dT;
  do
    {
      Vector<double> temperature = temperature_solver.get_temperature();

      apply_T_BC();
      temperature_solver.solve();

      temperature -= temperature_solver.get_temperature();
      max_dT = temperature.linfty_norm();

      std::cout << "max_dT=" << max_dT << " K\n";
    }
  while (max_dT > prm.get_double("Max temperature change"));

  temperature_solver.get_time_step() = dt0;
  prm_T.set("Max Newton iterations", n0);
  prm_T.set("Newton step length", a0);

  if (prm.get_bool("Approximate skin effect"))
    {
      approximate_LF_EM();
      prm.set("Approximate skin effect", false);
    }

  postprocess();
  output_results();
}

template <int dim>
void
Problem<dim>::solve_dislocation()
{
  std::cout << "Calculating dislocation density\n";

  initialize_dislocation();

  postprocess();
  output_results();

  while (true)
    {
      const bool keep_going = dislocation_solver.solve();

      if (!keep_going)
        break;
    };

  postprocess();
  output_results();
}

template <int dim>
void
Problem<dim>::solve_temperature_dislocation()
{
  std::cout << "Calculating temperature and dislocation density\n";

  initialize_dislocation();

  postprocess();
  output_results();

  for (unsigned int i = 1;; ++i)
    {
      update_min_time_step();

      const bool output_enabled = update_output_time_step(i);

      set_time_step(dislocation_solver.get_time_step());

      T_prev = temperature_solver.get_temperature();

      const int n_outer = prm.get_integer("Outer temperature iterations");

      bool keep_going_temp = false;

      for (int n = 0; n < n_outer; ++n)
        {
          apply_T_BC();
          keep_going_temp = temperature_solver.solve(n > 0);
        }

      dislocation_solver.get_temperature() =
        temperature_solver.get_temperature();

      const bool keep_going_disl = dislocation_solver.solve();

      postprocess();

      if (!keep_going_temp || !keep_going_disl)
        break;

      if (output_enabled)
        {
          output_results(false);

          std::cout << "Restoring previous dt=" << previous_time_step << "s\n";
          set_time_step(previous_time_step);
        }
    };

  output_results(false);
}

template <int dim>
void
Problem<dim>::solve_temperature()
{
  std::cout << "Calculating transient temperature field\n";

  postprocess();

  for (unsigned int i = 1;; ++i)
    {
      const bool output_enabled = update_output_time_step(i);

      T_prev = temperature_solver.get_temperature();

      const int n_outer = prm.get_integer("Outer temperature iterations");

      bool keep_going_temp = false;

      for (int n = 0; n < n_outer; ++n)
        {
          apply_T_BC();
          keep_going_temp = temperature_solver.solve(n > 0);
        }

      postprocess();

      if (!keep_going_temp)
        break;

      if (output_enabled)
        {
          output_results(false);

          std::cout << "Restoring previous dt=" << previous_time_step << "s\n";
          set_time_step(previous_time_step);
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
  const bool has_dislocation =
    with_dislocation() &&
    dislocation_solver.get_dof_handler().has_active_dofs();

  if (data)
    {
      temperature_solver.output_data();
      if (has_dislocation)
        dislocation_solver.output_data();
    }

  if (vtk)
    {
      temperature_solver.output_vtk();
      if (has_dislocation)
        dislocation_solver.output_vtk();
    }

  // Exports values at the crystal surface and on axis in 2D.
  // Export is disabled by default in 3D.
  if (boundary)
    for (unsigned int id = 0; id < 2; ++id)
      {
        temperature_solver.output_boundary_values(id);
        if (has_dislocation)
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
Problem<dim>::initialize_temperature()
{
  temperature_solver.initialize(); // sets T=0

  previous_time_step = temperature_solver.get_time_step();

  temperature_solver.output_mesh();

  dt_min_times = split_string(prm.get("Time step reduction instants"));

  const std::vector<double> X = split_string(prm.get("Probe coordinates x"));
  const std::vector<double> Z = split_string(prm.get("Probe coordinates z"));
  AssertDimension(X.size(), Z.size());

  for (unsigned int i = 0; i < X.size(); ++i)
    {
      Point<dim> p;
      p[0]       = X[i];
      p[dim - 1] = Z[i];
      temperature_solver.add_probe(p);
      dislocation_solver.add_probe(p);
    }

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Initial temperature"));

  if (dim == 3)
    {
      q3d.read_vtu("qEM-3d.vtu");
      q3d.convert(SurfaceInterpolator3D::CellField,
                  "QEM",
                  SurfaceInterpolator3D::PointField,
                  "q");
    }
  else if (dim == 2)
    {
      q2d.read_txt("qEM-2d.txt");
      T2d.read_txt("T-2d.txt");
    }

  const double t = temperature_solver.get_time();
  const double z = inductor_position->value(Point<1>(t));
  const double I = inductor_current->value(Point<1>(t));
  interpolate_q_em(z);
  previous_inductor_position = z;

  temperature_solver.add_output("I[A]", I);
  temperature_solver.add_output("z[m]", z);

  temperature_solver.output_parameter_table();

  if (prm.get_bool("Load saved results"))
    {
      temperature_solver.load_data();
      return;
    }
}

template <int dim>
void
Problem<dim>::initialize_dislocation()
{
  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature_solver.get_temperature();

  const double t = dislocation_solver.get_time();
  const double z = inductor_position->value(Point<1>(t));
  const double I = inductor_current->value(Point<1>(t));

  dislocation_solver.add_output("I[A]", I);
  dislocation_solver.add_output("z[m]", z);

  dislocation_solver.output_parameter_table();

  if (prm.get_bool("Load saved results"))
    {
      dislocation_solver.load_data();
    }

  if (dim == 2)
    dislocation_solver.get_stress_solver().set_bc1(boundary_id_axis, 0, 0);

  dislocation_solver.solve(true);

  previous_time_step = temperature_solver.get_time_step();
}

template <int dim>
void
Problem<dim>::apply_T_BC()
{
  const double t =
    temperature_solver.get_time() + temperature_solver.get_time_step();

  const double z = inductor_position->value(Point<1>(t));
  const double I = inductor_current->value(Point<1>(t));

  temperature_solver.add_output("z[m]", z);
  temperature_solver.add_output("I[A]", I);
  if (with_dislocation())
    {
      dislocation_solver.add_output("z[m]", z);
      dislocation_solver.add_output("I[A]", I);
    }

  if (T_BC1_applied())
    {
      std::vector<Point<dim>> points;
      std::vector<bool>       boundary_dofs;
      temperature_solver.get_boundary_points(boundary_id_surf,
                                             points,
                                             boundary_dofs);

      const double z0 = prm.get_double("Reference inductor position");
      const double dz = z0 - z;

      for (auto &p : points)
        p[dim - 1] += dz;

      const auto fields = T2d.get_field_names();

      std::map<double, std::string> T_time;
      std::vector<double>           times;

      for (const auto &f : fields)
        {
          if (f.substr(0, 7) == "T(K)@t=")
            {
              const double time = std::stod(f.substr(7));

              T_time[time] = f;
              times.push_back(time);
            }
        }

      // special handling of the steady-state data
      if (times.empty())
        {
          for (const auto &f : fields)
            {
              if (f == "T(K)")
                {
                  T_time[0] = f;
                  times.push_back(0);
                  break;
                }
            }
        }

      // interpolate and apply the BC
      const auto weights = get_interpolation_weights(times, t);

      Vector<double> T_BC(temperature_solver.get_temperature().size());

      for (const auto &it : weights)
        {
          Vector<double> tmp(T_BC);
          T2d.interpolate(T_time[it[0]], points, boundary_dofs, tmp);
          T_BC.add(it[1], tmp);
        }

      temperature_solver.set_bc1(boundary_id_surf, T_BC);

      return;
    }

  if (std::abs(z - previous_inductor_position) < 1e-12)
    std::cout << "z=const, skipping EM field interpolation\n";
  else
    {
      interpolate_q_em(z);
      previous_inductor_position = z;
    }

  Vector<double> q = q0;

  if (prm.get_bool("Use LF EM field"))
    {
      q = 0;
      approximate_LF_EM();
    }
  else
    {
      // apply the current and temperature-dependent electrical conductivity
      const Vector<double> &temperature = temperature_solver.get_temperature();

      const double I2 = sqr(I);
      for (unsigned int i = 0; i < q.size(); ++i)
        {
          const double s =
            electrical_conductivity.value(Point<1>(temperature[i]));
          q[i] *= I2 / std::sqrt(s);
        }
    }


  // Setting emissivity this way works but should be improved in the future
  const std::string e_expr = prm.get("Emissivity");
  const bool        e_T    = e_expr == "Ratnieks";
  const double      e_0    = e_T ? 0.46 : std::stod(e_expr);
  const double      T_0    = 1687;
  const double      T_a    = prm.get_double("Ambient temperature");

  std::function<double(const double)> emissivity_const = [=](const double) {
    return e_0;
  };

  std::function<double(const double)> emissivity_T = [=](const double T) {
    const double t = T / T_0;
    return e_0 * (t < 0.593 ? 1.39 : t > 1 ? 1 : 1.96 - 0.96 * t);
  };

  std::function<double(const double)> emissivity_deriv_const =
    [](const double) { return 0.0; };

  std::function<double(const double)> emissivity_deriv_T = [=](const double T) {
    const double t = T / T_0;
    return e_0 * (t < 0.593 || t > 1 ? 0.0 : -0.96 / T_0);
  };

  temperature_solver.set_bc_rad_mixed(boundary_id_surf,
                                      q,
                                      e_T ? emissivity_T : emissivity_const,
                                      e_T ? emissivity_deriv_T :
                                            emissivity_deriv_const,
                                      T_a);

  const double h     = prm.get_double("Heat transfer coefficient");
  const double T_ref = prm.get_double("Reference temperature");

  temperature_solver.set_bc_convective(boundary_id_surf, h, T_ref);
}

template <int dim>
void
Problem<dim>::interpolate_q_em(const double z)
{
  if (T_BC1_applied())
    {
      std::cout
        << "First-type temperature BC applied, skipping EM field interpolation\n";
      return;
    }

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  temperature_solver.get_boundary_points(boundary_id_surf,
                                         points,
                                         boundary_dofs);

  const double z0 = prm.get_double("Reference inductor position");
  const double dz = z0 - z;

  for (auto &p : points)
    p[dim - 1] += dz;

  if (prm.get_bool("Approximate skin effect") ||
      prm.get_bool("Use LF EM field"))
    std::fill(boundary_dofs.begin(), boundary_dofs.end(), true);

  if (dim == 2)
    q2d.interpolate("QEM", points, boundary_dofs, q0);
  else
    q3d.interpolate(
      SurfaceInterpolator3D::PointField, "q", points, boundary_dofs, q0);

  // normalize for future use
  q0 *= prm.get_double("QEM scale") *
        std::sqrt(prm.get_double("Reference electrical conductivity"));
}

template <int dim>
void
Problem<dim>::postprocess()
{
  update_T_max();
  update_tau_eff_max();
  measure_T();
}

template <int dim>
void
Problem<dim>::update_T_max()
{
  const Vector<double> &T = temperature_solver.get_temperature();

  const double dt = temperature_solver.get_time_step();

  if (T_max.size() == 0)
    {
      std::cout << "Initializing T_max\n";
      T_max = T;
    }

  if (T_prev.size() == 0)
    {
      std::cout << "Initializing T_prev\n";
      T_prev = T;
    }

  if (dot_T_min.size() == 0 || dot_T_max.size() == 0)
    {
      dot_T_min.reinit(T.size());
      dot_T_max.reinit(T.size());
    }

  AssertDimension(T_max.size(), T.size());
  AssertDimension(T_prev.size(), T.size());
  AssertDimension(dot_T_min.size(), T.size());
  AssertDimension(dot_T_max.size(), T.size());

  for (unsigned int k = 0; k < T_max.size(); ++k)
    {
      T_max[k] = std::max(T_max[k], T[k]);

      const double dot_T = (T[k] - T_prev[k]) / dt;

      dot_T_min[k] = std::min(dot_T_min[k], dot_T);
      dot_T_max[k] = std::max(dot_T_max[k], dot_T);
    }

  temperature_solver.add_field("T_max", T_max);
  temperature_solver.add_field("dot_T_min", dot_T_min);
  temperature_solver.add_field("dot_T_max", dot_T_max);

  if (with_dislocation())
    {
      dislocation_solver.add_field("T_max", T_max);
      dislocation_solver.add_field("dot_T_min", dot_T_min);
      dislocation_solver.add_field("dot_T_max", dot_T_max);
    }
}

template <int dim>
void
Problem<dim>::update_tau_eff_max()
{
  const bool has_dislocation =
    with_dislocation() &&
    dislocation_solver.get_dof_handler().has_active_dofs();

  if (!has_dislocation)
    return;

  Vector<double> tau;
  dislocation_solver.get_tau_eff(tau);

  if (tau_eff_max.size() == 0)
    {
      std::cout << "Initializing tau_eff_max\n";
      tau_eff_max = tau;
    }

  AssertDimension(tau_eff_max.size(), tau.size());

  for (unsigned int k = 0; k < tau_eff_max.size(); ++k)
    tau_eff_max[k] = std::max(tau_eff_max[k], tau[k]);

  dislocation_solver.add_field("tau_eff_max", tau_eff_max);
}

template <int dim>
void
Problem<dim>::measure_T()
{
  if (!inductor_probe_file.is_open())
    {
      const std::string s =
        "probes-inductor-temperature-" + std::to_string(dim) + "d.txt";

      std::cout << "Writing header to '" << s << "'\n";

      inductor_probe_file.open(s);

      for (unsigned int i = 0; i < inductor_probes.size(); ++i)
        inductor_probe_file << "# probe " << i << ":\t" << inductor_probes[i]
                            << "\n";

      inductor_probe_file
        << "t[s]\tdt[s]\twall_time[s]\tz[m]\tI[A]\tT_min[K]\tT_max[K]";

      for (unsigned int i = 0; i < inductor_probes.size(); ++i)
        inductor_probe_file << "\tT_" << i << "[K]";

      inductor_probe_file << "\n";
    }

  const double t     = temperature_solver.get_time();
  const double dt    = temperature_solver.get_time_step();
  const double I_ind = inductor_current->value(Point<1>(t));
  const double z_ind = inductor_position->value(Point<1>(t));

  const Vector<double> &temperature = temperature_solver.get_temperature();

  // Convert the vertical coordinate from inductor to crystal reference frame
  std::vector<Point<dim>> points = inductor_probes;
  for (auto &p : points)
    p[dim - 1] += z_ind;

  const std::vector<double> values_T =
    temperature_solver.get_field_at_points(temperature, points);

  const auto limits_T = minmax(temperature);

  const int precision =
    temperature_solver.get_parameters().get_integer("Output precision");
  inductor_probe_file << std::setprecision(precision);

  inductor_probe_file << t << '\t' << dt << '\t' << timer.wall_time() << '\t'
                      << z_ind << '\t' << I_ind << '\t' << limits_T.first
                      << '\t' << limits_T.second;

  for (const auto &v : values_T)
    inductor_probe_file << '\t' << v;

  inductor_probe_file << "\n";
}

template <int dim>
void
Problem<dim>::approximate_LF_EM()
{
  AssertThrow(dim == 2, ExcNotImplemented());

  const bool use_LF = prm.get_bool("Use LF EM field");

  std::vector<Point<dim>> points;
  std::vector<bool>       markers;
  temperature_solver.get_boundary_points(boundary_id_surf, points, markers);

  // calculate the radius
  double R = 0;
  for (unsigned int i = 0; i < points.size(); ++i)
    {
      if (markers[i])
        R = std::max(R, points[i][0]);
    }
  std::cout << "R=" << R << " m\n";

  // estimate the temperature change
  const double t = temperature_solver.get_time();
  const double I = inductor_current->value(Point<1>(t));
  const double f = prm.get_double("Inductor frequency");

  Vector<double> &temperature = temperature_solver.get_temperature();
  Vector<double>  temperature_deviation(temperature.size());
  Vector<double>  delta(temperature.size());
  Vector<double>  q = q0;

  for (unsigned int i = 0; i < q.size(); ++i)
    {
      const double s = electrical_conductivity.value(Point<1>(temperature[i]));

      // HF value
      q[i] *= sqr(I) / std::sqrt(s);
      delta[i] = 1.0 / std::sqrt(numbers::PI * 4e-7 * numbers::PI * f * s);

      const double a = 2 * q[i] / delta[i];
      const double x = R - points[i][0];

      const double lambda = temperature_solver.calc_lambda(temperature[i]);

      temperature_deviation[i] =
        -(a / lambda) * std::exp(-2 * x / delta[i]) * sqr(delta[i] / 2);

      // convert to LF
      q[i] *= 2 * std::exp(-2 * x / delta[i]) / delta[i];
    }

  if (use_LF)
    {
      temperature_solver.get_heat_source() = q;
    }
  else
    {
      std::cout << "Max T_deviation=" << temperature_deviation.linfty_norm()
                << " K\n";

      temperature_solver.add_field("T0", temperature);
      temperature += temperature_deviation;

      temperature_solver.add_field("T_deviation", temperature_deviation);
    }

  temperature_solver.add_field("q", q);
  temperature_solver.add_field("delta", delta);
}

template <int dim>
bool
Problem<dim>::T_BC1_applied() const
{
  return dim == 2 && !T2d.empty() > 0;
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
  int  dimension = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
      if (arguments[i] == "2d" || arguments[i] == "2D")
        dimension = 2;
      if (arguments[i] == "3d" || arguments[i] == "3D")
        dimension = 3;
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
