#ifndef macplas_dislocation_solver_h
#define macplas_dislocation_solver_h

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <iostream>

#include "stress_solver.h"

using namespace dealii;

/// Class for calculation of the time-dependent dislocation density
template <int dim>
class DislocationSolver
{
public:
  /// Constructor

  /// Initialize the solver parameters from \c dislocation.prm.
  /// If it doesn't exist, the default parameter values are written to
  /// \c dislocation-default.prm.
  DislocationSolver(const unsigned int order = 2);

  /// Calculate the dislocation density and creep strain
  bool
  solve();

  /// Get mesh
  const Triangulation<dim> &
  get_mesh() const;
  /// Get mesh
  Triangulation<dim> &
  get_mesh();

  /// Get temperature
  const Vector<double> &
  get_temperature() const;
  /// Get temperature
  Vector<double> &
  get_temperature();

  /// Get dislocation density
  const Vector<double> &
  get_dislocation_density() const;
  /// Get dislocation density
  Vector<double> &
  get_dislocation_density();

  /// Get stress
  const BlockVector<double> &
  get_stress() const;
  /// Get stress deviator
  const BlockVector<double> &
  get_stress_deviator() const;
  /// Get second invariant of deviatoric stress
  const Vector<double> &
  get_stress_J_2() const;

  /// Get creep strain
  const BlockVector<double> &
  get_strain_c() const;
  /// Get creep strain
  BlockVector<double> &
  get_strain_c();

  /// Get degrees of freedom for temperature
  const DoFHandler<dim> &
  get_dof_handler() const;

  /// Get stress solver (for setting boundary conditions)
  StressSolver<dim> &
  get_stress_solver();

  /// Current time, s
  double
  get_time() const;
  /// Time step, s
  double
  get_time_step() const;
  /// Final time, s
  double
  get_max_time() const;

  /// Initialize fields
  void
  initialize();

  /// Add probe point
  void
  add_probe(const Point<dim> &p);

  /// Save results to disk
  void
  output_results() const;

  /// Save mesh to disk
  void
  output_mesh() const;

private:
  /// Initialize parameters. Called by the constructor
  void
  initialize_parameters();

  /// Write current values of fields at probe points to disk
  void
  output_probes() const;

  /// Evaluate values of source field at probe points
  std::vector<double>
  get_field_at_probes(const Vector<double> &source) const;

  /// Calculate the effective stress \f$\tau_\mathrm{eff}\f$
  double
  tau_eff(const double N_m, const double J_2) const;
  /// Same as above, for vector arguments
  std::vector<double>
  tau_eff(const std::vector<double> &N_m, const std::vector<double> &J_2) const;
  /// Same as above, for Vector arguments
  Vector<double>
  tau_eff(const Vector<double> &N_m, const Vector<double> &J_2) const;

  /// Calculate the time derivative of dislocation density \f$\dot{N_m}\f$
  double
  derivative_N_m(const double N_m, const double J_2, const double T) const;
  /// Same as above, for vector arguments
  std::vector<double>
  derivative_N_m(const std::vector<double> &N_m,
                 const std::vector<double> &J_2,
                 const std::vector<double> &T) const;
  /// Same as above, for Vector arguments
  Vector<double>
  derivative_N_m(const Vector<double> &N_m,
                 const Vector<double> &J_2,
                 const Vector<double> &T) const;

  /// Calculate the partial derivative \f$\partial \dot{N_m} / \partial N_m\f$
  double
  derivative2_N_m_N_m(const double N_m, const double J_2, const double T) const;

  /// Calculate the creep strain rate \f$\dot{\varepsilon^c_{ij}}\f$
  double
  derivative_strain(const double N_m,
                    const double J_2,
                    const double T,
                    const double S) const;
  /// Same as above, for vector arguments
  std::vector<double>
  derivative_strain(const std::vector<double> &N_m,
                    const std::vector<double> &J_2,
                    const std::vector<double> &T,
                    const std::vector<double> &S) const;
  /// Same as above, for Vector arguments
  Vector<double>
  derivative_strain(const Vector<double> &N_m,
                    const Vector<double> &J_2,
                    const Vector<double> &T,
                    const Vector<double> &S) const;

  /// Calculate the dislocation velocity \f$v\f$
  double
  dislocation_velocity(const double N_m,
                       const double J_2,
                       const double T) const;
  /// Same as above, for vector arguments
  std::vector<double>
  dislocation_velocity(const std::vector<double> &N_m,
                       const std::vector<double> &J_2,
                       const std::vector<double> &T) const;
  /// Same as above, for Vector arguments
  Vector<double>
  dislocation_velocity(const Vector<double> &N_m,
                       const Vector<double> &J_2,
                       const Vector<double> &T) const;

  /// Time integration using the forward Euler method (explicit)
  void
  integrate_Euler();

  /// Time integration using the midpoint method (explicit), also known as RK2
  void
  integrate_midpoint();

  /// Time integration using analytical expression for linearized N_m
  void
  integrate_linearized_N_m();

  /// Time integration using analytical expression for linearized N_m but the
  /// midpoint method for creep strain
  void
  integrate_linearized_N_m_midpoint();

  /// Stress solver

  /// To avoid redundancy, mesh, finite element, degree handler and temperature
  /// field is not stored in DislocationSolver but in StressSolver
  StressSolver<dim> stress_solver;

  Vector<double> dislocation_density; ///< Dislocation density, m<sup>-2</sup>

  std::vector<Point<dim>> probes; ///< Locations of probe points

  ParameterHandler prm; ///< Parameter handler

  std::string time_scheme; ///< Time integration scheme

  int current_step; ///< Time stepping: index of the current time step

  double m_b; ///< Magnitude of Burgers vector \f$b\f$, m
  double m_Q; ///< Peierls potential \f$Q\f$, eV
  double m_D; ///< Strain hardening factor \f$D\f$, N m<sup>-1</sup>
  double m_K; ///< Material constant \f$K\f$, m N<sup>-1</sup>

  /// Material constant \f$k_0, \text{m}^{2p+l}\;\text{N}^{-p}\;\text{s}^{-1}\f$
  double m_k_0;
  double m_l; ///< Material constant \f$l\f$, -
  double m_p; ///< Material constant \f$p\f$, -

  /// Boltzmann constant \f$k_B\f$, eV/K
  static constexpr double m_k_B = 8.617e-5;
};

template <int dim>
DislocationSolver<dim>::DislocationSolver(const unsigned int order)
  : stress_solver(order)
  , current_step(0)
{
  // Physical parameters from https://doi.org/10.1016/j.jcrysgro.2016.05.027
  prm.declare_entry("Burgers vector",
                    "3.8e-10",
                    Patterns::Double(),
                    "Magnitude of Burgers vector in m");

  prm.declare_entry("Peierls potential",
                    "2.17",
                    Patterns::Double(),
                    "Peierls potential in eV");

  prm.declare_entry("Strain hardening factor",
                    "4.3",
                    Patterns::Double(),
                    "Strain hardening factor in N/m");

  prm.declare_entry("Material constant K",
                    "3.1e-4",
                    Patterns::Double(),
                    "Material constant K in m/N");

  prm.declare_entry("Material constant k_0",
                    "8.6e-4",
                    Patterns::Double(),
                    "Material constant k_0 in m^(2p+l)/N^p/s");

  prm.declare_entry("Material constant l",
                    "1.0",
                    Patterns::Double(),
                    "Material constant l (dimensionless)");

  prm.declare_entry("Material constant p",
                    "1.1",
                    Patterns::Double(),
                    "Material constant p (dimensionless)");


  prm.declare_entry("Time step",
                    "1",
                    Patterns::Double(),
                    "Time step in seconds (0 - steady-state)");

  prm.declare_entry("Max time",
                    "10",
                    Patterns::Double(),
                    "Max time in seconds");

  prm.declare_entry("Time scheme",
                    "Euler",
                    Patterns::Anything(),
                    "Time integration scheme");


  prm.declare_entry("Output precision",
                    "8",
                    Patterns::Integer(1),
                    "Precision of double variables for output of field data");

  try
    {
      prm.parse_input("dislocation.prm");
    }
  catch (std::exception &e)
    {
      std::cout << e.what() << "\n";

      std::ofstream of("dislocation-default.prm");
      prm.print_parameters(of, ParameterHandler::Text);
    }

  initialize_parameters();
}

template <int dim>
bool
DislocationSolver<dim>::solve()
{
  current_step++;
  const double dt    = get_time_step();
  const double t     = get_time();
  const double t_max = get_max_time();

  std::cout << "Time " << t << " s\n";

  // first time step: calculate stresses
  if (current_step == 1)
    stress_solver.solve();

  if (time_scheme == "Euler")
    integrate_Euler();
  else if (time_scheme == "Midpoint" || time_scheme == "RK2")
    integrate_midpoint();
  else if (time_scheme == "Linearized N_m")
    integrate_linearized_N_m();
  else if (time_scheme == "Linearized N_m midpoint" ||
           time_scheme == "Linearized N_m RK2")
    integrate_linearized_N_m_midpoint();
  else
    AssertThrow(false, ExcNotImplemented());

  output_probes();

  if (dt > 0 && t >= t_max)
    return false;

  return dt > 0;
}

template <int dim>
const Triangulation<dim> &
DislocationSolver<dim>::get_mesh() const
{
  return stress_solver.get_mesh();
}

template <int dim>
Triangulation<dim> &
DislocationSolver<dim>::get_mesh()
{
  return stress_solver.get_mesh();
}

template <int dim>
const Vector<double> &
DislocationSolver<dim>::get_temperature() const
{
  return stress_solver.get_temperature();
}

template <int dim>
Vector<double> &
DislocationSolver<dim>::get_temperature()
{
  return stress_solver.get_temperature();
}

template <int dim>
const Vector<double> &
DislocationSolver<dim>::get_dislocation_density() const
{
  return dislocation_density;
}

template <int dim>
Vector<double> &
DislocationSolver<dim>::get_dislocation_density()
{
  return dislocation_density;
}

template <int dim>
const BlockVector<double> &
DislocationSolver<dim>::get_stress() const
{
  return stress_solver.get_stress();
}

template <int dim>
const BlockVector<double> &
DislocationSolver<dim>::get_stress_deviator() const
{
  return stress_solver.get_stress_deviator();
}

template <int dim>
const Vector<double> &
DislocationSolver<dim>::get_stress_J_2() const
{
  return stress_solver.get_stress_J_2();
}

template <int dim>
const BlockVector<double> &
DislocationSolver<dim>::get_strain_c() const
{
  return stress_solver.get_strain_c();
}

template <int dim>
BlockVector<double> &
DislocationSolver<dim>::get_strain_c()
{
  return stress_solver.get_strain_c();
}

template <int dim>
const DoFHandler<dim> &
DislocationSolver<dim>::get_dof_handler() const
{
  return stress_solver.get_dof_handler();
}

template <int dim>
StressSolver<dim> &
DislocationSolver<dim>::get_stress_solver()
{
  return stress_solver;
}

template <int dim>
double
DislocationSolver<dim>::get_time() const
{
  return get_time_step() * current_step;
}

template <int dim>
double
DislocationSolver<dim>::get_time_step() const
{
  return prm.get_double("Time step");
}

template <int dim>
double
DislocationSolver<dim>::get_max_time() const
{
  return prm.get_double("Max time");
}

template <int dim>
void
DislocationSolver<dim>::initialize()
{
  stress_solver.initialize(); // prints info

  const unsigned int n_dofs_temp = get_temperature().size();

  dislocation_density.reinit(n_dofs_temp);
}

template <int dim>
void
DislocationSolver<dim>::add_probe(const Point<dim> &p)
{
  probes.push_back(p);
}

template <int dim>
void
DislocationSolver<dim>::output_results() const
{
  Timer timer;

  const double t = get_time();

  const DoFHandler<dim> &   dh = get_dof_handler();
  const FiniteElement<dim> &fe = dh.get_fe();

  std::stringstream ss;
  ss << "result-" << dim << "d-order" << fe.degree << "-t" << t << ".vtk";
  const std::string file_name = ss.str();
  std::cout << "Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh);

  const Vector<double> &T   = get_temperature();
  const Vector<double> &N_m = get_dislocation_density();

  data_out.add_data_vector(T, "T");
  data_out.add_data_vector(N_m, "N_m");

  const BlockVector<double> &stress = get_stress();
  for (unsigned int i = 0; i < stress.n_blocks(); ++i)
    {
      const std::string name = "stress_" + std::to_string(i);
      data_out.add_data_vector(stress.block(i), name);
    }

  const BlockVector<double> &stress_deviator = get_stress_deviator();
  for (unsigned int i = 0; i < stress_deviator.n_blocks(); ++i)
    {
      const std::string name = "stress_deviator_" + std::to_string(i);
      data_out.add_data_vector(stress_deviator.block(i), name);
    }

  const Vector<double> &stress_J_2 = get_stress_J_2();
  data_out.add_data_vector(get_stress_J_2(), "stress_J_2");

  const unsigned int n_dofs_temp = T.size();

  const Vector<double> tau     = tau_eff(N_m, stress_J_2);
  const Vector<double> dot_N_m = derivative_N_m(N_m, stress_J_2, T);
  const Vector<double> v       = dislocation_velocity(N_m, stress_J_2, T);

  BlockVector<double> dot_epsilon_c(StressSolver<dim>::n_components);
  for (unsigned int i = 0; i < dot_epsilon_c.n_blocks(); ++i)
    {
      dot_epsilon_c.block(i) =
        derivative_strain(N_m, stress_J_2, T, stress_deviator.block(i));
    }

  data_out.add_data_vector(tau, "tau_eff");
  data_out.add_data_vector(dot_N_m, "dot_N_m");
  data_out.add_data_vector(v, "v");
  for (unsigned int i = 0; i < dot_epsilon_c.n_blocks(); ++i)
    {
      const std::string name = "dot_epsilon_c_" + std::to_string(i);
      data_out.add_data_vector(dot_epsilon_c.block(i), name);
    }

  data_out.build_patches(fe.degree);

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  data_out.write_vtk(output);

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
void
DislocationSolver<dim>::output_mesh() const
{
  Timer timer;

  std::stringstream ss;
  ss << "mesh-" << dim << "d-order" << get_dof_handler().get_fe().degree
     << ".msh";
  const std::string file_name = ss.str();
  std::cout << "Saving to '" << file_name << "'";

  std::ofstream output(file_name);

  GridOut grid_out;
  grid_out.set_flags(GridOutFlags::Msh(true));
  grid_out.write_msh(get_mesh(), output);

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
void
DislocationSolver<dim>::initialize_parameters()
{
  std::cout << "Intializing parameters";

  m_b   = prm.get_double("Burgers vector");
  m_Q   = prm.get_double("Peierls potential");
  m_D   = prm.get_double("Strain hardening factor");
  m_K   = prm.get_double("Material constant K");
  m_k_0 = prm.get_double("Material constant k_0");
  m_l   = prm.get_double("Material constant l");
  m_p   = prm.get_double("Material constant p");

  time_scheme = prm.get("Time scheme");

  std::cout << "  done\n";

  std::cout << "b=" << m_b << "\n"
            << "Q=" << m_Q << "\n"
            << "D=" << m_D << "\n"
            << "K=" << m_K << "\n"
            << "k_0=" << m_k_0 << "\n"
            << "l=" << m_l << "\n"
            << "p=" << m_p << "\n"
            << "k_B=" << m_k_B << "\n"
            << "time_scheme=" << time_scheme << "\n";
}

template <int dim>
void
DislocationSolver<dim>::output_probes() const
{
  Timer timer;

  std::stringstream ss;
  ss << "probes-" << dim << "d.txt";
  const std::string file_name = ss.str();

  std::cout << "Saving values at probe points to '" << file_name << "'";

  const unsigned int N = probes.size();

  const double t = get_time();

  const BlockVector<double> &s   = get_stress();
  const BlockVector<double> &e_c = get_strain_c();

  if (current_step == 1)
    {
      // write header at the first time step
      std::ofstream output(file_name);

      for (unsigned int i = 0; i < N; ++i)
        output << "# probe " << i << ": " << probes[i] << "\n";

      // Write column titles, avoiding spaces in the units.
      // In future, consider using tabs as separators.
      output << "t[s]";
      for (unsigned int i = 0; i < N; ++i)
        {
          output << " T_" << i << "[K]"
                 << " N_m_" << i << "[m^-2]"
                 << " dot_N_m_" << i << "[m^-2s^-1]"
                 << " v_" << i << "[ms^-1]";

          for (unsigned int j = 0; j < s.n_blocks(); ++j)
            output << " stress_" << j << "_" << i << "[Pa]";

          for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
            output << " strain_c_" << j << "_" << i << "[-]";

          for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
            output << " dot_strain_c_" << j << "_" << i << "[s^-1]";

          output << " tau_eff_" << i << "[Pa]"
                 << " J_2_" << i << "[Pa^2]";
        }
      output << "\n";
    }

  const Vector<double> &T   = get_temperature();
  const Vector<double> &N_m = get_dislocation_density();
  const Vector<double> &J_2 = get_stress_J_2();

  // Process each field separately - easy to implement but potentially slow.
  const std::vector<double> values_T   = get_field_at_probes(T);
  const std::vector<double> values_N_m = get_field_at_probes(N_m);
  const std::vector<double> values_J_2 = get_field_at_probes(J_2);

  std::vector<std::vector<double>> values_s(s.n_blocks());
  for (unsigned int i = 0; i < s.n_blocks(); ++i)
    values_s[i] = get_field_at_probes(s.block(i));

  std::vector<std::vector<double>> values_e_c(e_c.n_blocks());
  for (unsigned int i = 0; i < e_c.n_blocks(); ++i)
    values_e_c[i] = get_field_at_probes(e_c.block(i));

  // calculate additional fields
  const std::vector<double> values_tau = tau_eff(values_N_m, values_J_2);

  const std::vector<double> values_dot_N_m =
    derivative_N_m(values_N_m, values_J_2, values_T);

  const std::vector<double> values_v =
    dislocation_velocity(values_N_m, values_J_2, values_T);

  std::vector<std::vector<double>> values_dot_e_c(e_c.n_blocks());
  for (unsigned int i = 0; i < e_c.n_blocks(); ++i)
    values_dot_e_c[i] =
      derivative_strain(values_N_m, values_J_2, values_T, values_s[i]);

  // header is already written, append values at the current time step
  std::ofstream output(file_name, std::ios::app);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  output << t;
  for (unsigned int i = 0; i < N; ++i)
    {
      output << " " << values_T[i] << " " << values_N_m[i] << " "
             << values_dot_N_m[i] << " " << values_v[i];

      for (unsigned int j = 0; j < s.n_blocks(); ++j)
        output << " " << values_s[j][i];

      for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
        output << " " << values_e_c[j][i];

      for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
        output << " " << values_dot_e_c[j][i];

      output << " " << values_tau[i] << " " << values_J_2[i];
    }
  output << "\n";

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
std::vector<double>
DislocationSolver<dim>::get_field_at_probes(const Vector<double> &source) const
{
  Functions::FEFieldFunction<dim> ff(get_dof_handler(), source);

  std::vector<double> values(probes.size());
  ff.value_list(probes, values);

  return values;
}

template <int dim>
double
DislocationSolver<dim>::tau_eff(const double N_m, const double J_2) const
{
  return std::max(std::sqrt(J_2) - m_D * std::sqrt(N_m), 0.0);
}

template <int dim>
std::vector<double>
DislocationSolver<dim>::tau_eff(const std::vector<double> &N_m,
                                const std::vector<double> &J_2) const
{
  const unsigned int N = N_m.size();

  std::vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = tau_eff(N_m[i], J_2[i]);

  return result;
}

template <int dim>
Vector<double>
DislocationSolver<dim>::tau_eff(const Vector<double> &N_m,
                                const Vector<double> &J_2) const
{
  const unsigned int N = N_m.size();

  Vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = tau_eff(N_m[i], J_2[i]);

  return result;
}

template <int dim>
double
DislocationSolver<dim>::derivative_N_m(const double N_m,
                                       const double J_2,
                                       const double T) const
{
  const double tau = tau_eff(N_m, J_2);

  return m_K * m_k_0 * std::pow(tau, m_p + m_l) * std::exp(-m_Q / (m_k_B * T)) *
         N_m;
}

template <int dim>
std::vector<double>
DislocationSolver<dim>::derivative_N_m(const std::vector<double> &N_m,
                                       const std::vector<double> &J_2,
                                       const std::vector<double> &T) const
{
  const unsigned int N = N_m.size();

  std::vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = derivative_N_m(N_m[i], J_2[i], T[i]);

  return result;
}

template <int dim>
Vector<double>
DislocationSolver<dim>::derivative_N_m(const Vector<double> &N_m,
                                       const Vector<double> &J_2,
                                       const Vector<double> &T) const
{
  const unsigned int N = N_m.size();

  Vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = derivative_N_m(N_m[i], J_2[i], T[i]);

  return result;
}

template <int dim>
double
DislocationSolver<dim>::derivative2_N_m_N_m(const double N_m,
                                            const double J_2,
                                            const double T) const
{
  const double tau = tau_eff(N_m, J_2);

  return m_K * m_k_0 * std::exp(-m_Q / (m_k_B * T)) *
         (std::pow(tau, m_p + m_l) - (m_p + m_l) *
                                       std::pow(tau, m_p + m_l - 1) *
                                       std::sqrt(N_m) * m_D / 2);
}

template <int dim>
double
DislocationSolver<dim>::derivative_strain(const double N_m,
                                          const double J_2,
                                          const double T,
                                          const double S) const
{
  const double tau = tau_eff(N_m, J_2);

  return m_b * m_k_0 * N_m * std::pow(tau, m_p) * std::exp(-m_Q / (m_k_B * T)) *
         S / (2 * std::sqrt(J_2));
}

template <int dim>
std::vector<double>
DislocationSolver<dim>::derivative_strain(const std::vector<double> &N_m,
                                          const std::vector<double> &J_2,
                                          const std::vector<double> &T,
                                          const std::vector<double> &S) const
{
  const unsigned int N = N_m.size();

  std::vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = derivative_strain(N_m[i], J_2[i], T[i], S[i]);

  return result;
}

template <int dim>
Vector<double>
DislocationSolver<dim>::derivative_strain(const Vector<double> &N_m,
                                          const Vector<double> &J_2,
                                          const Vector<double> &T,
                                          const Vector<double> &S) const
{
  const unsigned int N = N_m.size();

  Vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = derivative_strain(N_m[i], J_2[i], T[i], S[i]);

  return result;
}

template <int dim>
double
DislocationSolver<dim>::dislocation_velocity(const double N_m,
                                             const double J_2,
                                             const double T) const
{
  const double tau = tau_eff(N_m, J_2);

  return m_k_0 * std::pow(tau, m_p) * std::exp(-m_Q / (m_k_B * T));
}

template <int dim>
std::vector<double>
DislocationSolver<dim>::dislocation_velocity(const std::vector<double> &N_m,
                                             const std::vector<double> &J_2,
                                             const std::vector<double> &T) const
{
  const unsigned int N = N_m.size();

  std::vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = dislocation_velocity(N_m[i], J_2[i], T[i]);

  return result;
}

template <int dim>
Vector<double>
DislocationSolver<dim>::dislocation_velocity(const Vector<double> &N_m,
                                             const Vector<double> &J_2,
                                             const Vector<double> &T) const
{
  const unsigned int N = N_m.size();

  Vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = dislocation_velocity(N_m[i], J_2[i], T[i]);

  return result;
}

template <int dim>
void
DislocationSolver<dim>::integrate_Euler()
{
  // Implementation of explicit time integration. In future, consider using
  // deal.II functionality (TimeStepping::RungeKutta)

  Vector<double> &     N_m       = get_dislocation_density();
  BlockVector<double> &epsilon_c = get_strain_c();

  const Vector<double> &     T   = get_temperature();
  const Vector<double> &     J_2 = get_stress_J_2();
  const BlockVector<double> &S   = get_stress_deviator();

  const unsigned int N  = N_m.size();
  const double       dt = get_time_step();

  for (unsigned int i = 0; i < N; ++i)
    {
      // update strains
      for (unsigned int j = 0; j < epsilon_c.n_blocks(); ++j)
        epsilon_c.block(j)[i] +=
          derivative_strain(N_m[i], J_2[i], T[i], S.block(j)[i]) * dt;

      // update N_m
      N_m[i] += derivative_N_m(N_m[i], J_2[i], T[i]) * dt;
    }

  stress_solver.solve();
}

template <int dim>
void
DislocationSolver<dim>::integrate_midpoint()
{
  Vector<double> &     N_m       = get_dislocation_density();
  BlockVector<double> &epsilon_c = get_strain_c();

  const Vector<double> &     T   = get_temperature();
  const Vector<double> &     J_2 = get_stress_J_2();
  const BlockVector<double> &S   = get_stress_deviator();

  const unsigned int N  = N_m.size();
  const double       dt = get_time_step();

  // save values at the beginning of time step
  Vector<double>      N_m_0       = N_m;
  BlockVector<double> epsilon_c_0 = epsilon_c;

  // first, take a half step
  for (unsigned int i = 0; i < N; ++i)
    {
      // update strains
      for (unsigned int j = 0; j < epsilon_c.n_blocks(); ++j)
        epsilon_c.block(j)[i] +=
          derivative_strain(N_m[i], J_2[i], T[i], S.block(j)[i]) * dt / 2;

      // update N_m
      N_m[i] += derivative_N_m(N_m[i], J_2[i], T[i]) * dt / 2;
    }

  // recalculate stresses
  stress_solver.solve();

  // now, take a full step with derivatives evaluated at the midpoint
  for (unsigned int i = 0; i < N; ++i)
    {
      // update strains
      for (unsigned int j = 0; j < epsilon_c.n_blocks(); ++j)
        epsilon_c.block(j)[i] =
          epsilon_c_0.block(j)[i] +
          derivative_strain(N_m[i], J_2[i], T[i], S.block(j)[i]) * dt;

      // update N_m
      N_m[i] = N_m_0[i] + derivative_N_m(N_m[i], J_2[i], T[i]) * dt;
    }

  stress_solver.solve();
}

template <int dim>
void
DislocationSolver<dim>::integrate_linearized_N_m()
{
  Vector<double> &     N_m       = get_dislocation_density();
  BlockVector<double> &epsilon_c = get_strain_c();

  const Vector<double> &     T   = get_temperature();
  const Vector<double> &     J_2 = get_stress_J_2();
  const BlockVector<double> &S   = get_stress_deviator();

  const unsigned int N  = N_m.size();
  const double       dt = get_time_step();

  for (unsigned int i = 0; i < N; ++i)
    {
      // linearize dot_N_m = a + b * (N_m-N_m_0)
      const double a = derivative_N_m(N_m[i], J_2[i], T[i]);
      const double b = derivative2_N_m_N_m(N_m[i], J_2[i], T[i]);

      // integrate analytically, assuming constant stresses
      const double d_N_m = a / b * (std::exp(b * dt) - 1);

      // update N_m
      N_m[i] += d_N_m;

      // update strains
      for (unsigned int j = 0; j < epsilon_c.n_blocks(); ++j)
        epsilon_c.block(j)[i] +=
          derivative_strain(N_m[i], J_2[i], T[i], S.block(j)[i]) * dt;
    }

  stress_solver.solve();
}

template <int dim>
void
DislocationSolver<dim>::integrate_linearized_N_m_midpoint()
{
  Vector<double> &     N_m       = get_dislocation_density();
  BlockVector<double> &epsilon_c = get_strain_c();

  const Vector<double> &     T   = get_temperature();
  const Vector<double> &     J_2 = get_stress_J_2();
  const BlockVector<double> &S   = get_stress_deviator();

  const unsigned int N  = N_m.size();
  const double       dt = get_time_step();

  // save values at the beginning of time step
  Vector<double>      N_m_0       = N_m;
  BlockVector<double> epsilon_c_0 = epsilon_c;

  // first, take a half step
  for (unsigned int i = 0; i < N; ++i)
    {
      // update strains
      for (unsigned int j = 0; j < epsilon_c.n_blocks(); ++j)
        epsilon_c.block(j)[i] +=
          derivative_strain(N_m[i], J_2[i], T[i], S.block(j)[i]) * dt / 2;

      // linearize dot_N_m = a + b * (N_m-N_m_0)
      const double a = derivative_N_m(N_m[i], J_2[i], T[i]);
      const double b = derivative2_N_m_N_m(N_m[i], J_2[i], T[i]);

      // integrate analytically, assuming constant stresses
      N_m[i] += a / b * (std::exp(b * dt / 2) - 1);
    }

  // recalculate stresses
  stress_solver.solve();

  // now, take a full step with derivatives evaluated at the midpoint
  for (unsigned int i = 0; i < N; ++i)
    {
      // update strains
      for (unsigned int j = 0; j < epsilon_c.n_blocks(); ++j)
        epsilon_c.block(j)[i] =
          epsilon_c_0.block(j)[i] +
          derivative_strain(N_m[i], J_2[i], T[i], S.block(j)[i]) * dt;

      // linearize dot_N_m = a + b * (N_m-N_m_0)
      const double a = derivative_N_m(N_m_0[i], J_2[i], T[i]);
      const double b = derivative2_N_m_N_m(N_m_0[i], J_2[i], T[i]);

      // update N_m
      N_m[i] = N_m_0[i] + a / b * (std::exp(b * dt) - 1);
    }

  stress_solver.solve();
}

#endif