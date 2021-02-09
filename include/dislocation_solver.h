#ifndef macplas_dislocation_solver_h
#define macplas_dislocation_solver_h

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <iostream>

#include "stress_solver.h"
#include "utilities.h"

using namespace dealii;

/** Class for calculation of the time-dependent dislocation density
 */
template <int dim>
class DislocationSolver
{
public:
  /** Constructor.
   * Initialize the solver parameters from \c dislocation.prm.
   * If it doesn't exist, the default parameter values are written to
   * \c dislocation-default.prm.
   */
  DislocationSolver(const unsigned int order = 2);

  /** Advance time and calculate the dislocation density and creep strain.
   * Calls \c DislocationSolver::advance_time unless \c stress_only is enabled,
   * then only the stress is calculated.
   */
  bool
  solve(const double stress_only = false);

  /** Get mesh
   */
  const Triangulation<dim> &
  get_mesh() const;

  /** Get mesh
   */
  Triangulation<dim> &
  get_mesh();

  /** Get temperature \f$T\f$, K
   */
  const Vector<double> &
  get_temperature() const;

  /** Get temperature \f$T\f$, K
   */
  Vector<double> &
  get_temperature();

  /** Get dislocation density \f$N_m\f$, m<sup>-2</sup>
   */
  const Vector<double> &
  get_dislocation_density() const;

  /** Get dislocation density \f$N_m\f$, m<sup>-2</sup>
   */
  Vector<double> &
  get_dislocation_density();

  /** Get displacement \f$\mathbf{u}\f$, m.
   * Calls \c StressSolver::get_displacement
   */
  const BlockVector<double> &
  get_displacement() const;

  /** Get displacement \f$\mathbf{u}\f$, m.
   * Calls \c StressSolver::get_displacement
   */
  BlockVector<double> &
  get_displacement();

  /** Get stress \f$\sigma_{ij}\f$, Pa.
   * Calls \c StressSolver::get_stress
   */
  const BlockVector<double> &
  get_stress() const;

  /** Get stress deviator \f$S_{ij}\f$, Pa.
   * Calls \c StressSolver::get_stress_deviator
   */
  const BlockVector<double> &
  get_stress_deviator() const;

  /** Get second invariant of deviatoric stress \f$J_2\f$, Pa<sup>2</sup>.
   * Calls \c StressSolver::get_stress_J_2
   */
  const Vector<double> &
  get_stress_J_2() const;

  /** Get creep strain \f$\varepsilon^c_{ij}\f$, dimensionless
   */
  const BlockVector<double> &
  get_strain_c() const;

  /** Get creep strain \f$\varepsilon^c_{ij}\f$, dimensionless
   */
  BlockVector<double> &
  get_strain_c();

  /** Get degrees of freedom for temperature
   */
  const DoFHandler<dim> &
  get_dof_handler() const;

  /** Get stress solver (for setting boundary conditions)
   */
  StressSolver<dim> &
  get_stress_solver();

  /** Current time \f$t\f$, s
   */
  double
  get_time() const;

  /** Current time \f$t\f$, s
   */
  double &
  get_time();

  /** Current time step \f$\Delta t\f$, s
   */
  double
  get_time_step() const;

  /** Current time step \f$\Delta t\f$, s
   */
  double &
  get_time_step();

  /** Final time, s
   */
  double
  get_max_time() const;

  /** Initialize fields
   */
  void
  initialize();

  /** Add probe point
   */
  void
  add_probe(const Point<dim> &p);

  /** Read raw results from disk
   */
  void
  load_data();

  /** Save raw results to disk
   */
  void
  output_data() const;

  /** Save results to disk in \c vtk format
   */
  void
  output_vtk() const;

  /** Save mesh to disk in \c msh format
   */
  void
  output_mesh() const;

private:
  /** Initialize parameters. Called by the constructor
   */
  void
  initialize_parameters();

  /** Minimum time step \f$\Delta t_\min\f$, s
   */
  double
  get_time_step_min() const;

  /** Maximum time step \f$\Delta t_\max\f$, s
   */
  double
  get_time_step_max() const;

  /** Time stepping: limit time step according to minimum and maximum values
   */
  void
  limit_time_step();

  /** Time stepping: adjust time step according to user-specified settings.
   * Considers maximum \f$v \Delta t\f$, \f$\Delta \varepsilon^c\f$ and relative
   * \f$\Delta N_m\f$, calls \c DislocationSolver::limit_time_step.
   */
  void
  update_time_step();

  /** Time stepping: advance time \f$t \to t + \Delta t\f$
   */
  void
  advance_time();

  /** Write current values of fields at probe points to disk.
   * File name \c "probes-dislocation-<dim>d.txt"
   */
  void
  output_probes() const;

  /** Evaluate values of source field at probe points
   */
  std::vector<double>
  get_field_at_probes(const Vector<double> &source) const;

  /** Calculate the temperature-dependent Peierls potential \f$Q\f$
   */
  double
  calc_Q(const double T) const;

  /** Calculate the temperature-dependent strain hardening factor \f$D\f$
   */
  double
  calc_D(const double T) const;

  /** Calculate the effective stress \f$\tau_\mathrm{eff} =
   * \left\langle S \sqrt{J_2} - D \sqrt{N_m} \right\rangle\f$
   */
  double
  tau_eff(const double N_m, const double J_2, const double T) const;

  /** Same as above, for \c vector arguments
   */
  std::vector<double>
  tau_eff(const std::vector<double> &N_m,
          const std::vector<double> &J_2,
          const std::vector<double> &T) const;

  /** Same as above, for \c Vector arguments
   */
  Vector<double>
  tau_eff(const Vector<double> &N_m,
          const Vector<double> &J_2,
          const Vector<double> &T) const;

  /** Calculate the time derivative of dislocation density \f$\dot{N_m} =
   * K v \tau_\mathrm{eff}^l N_m\f$
   */
  double
  derivative_N_m(const double N_m, const double J_2, const double T) const;

  /** Same as above, for \c vector arguments
   */
  std::vector<double>
  derivative_N_m(const std::vector<double> &N_m,
                 const std::vector<double> &J_2,
                 const std::vector<double> &T) const;
  /** Same as above, for \c Vector arguments
   */
  Vector<double>
  derivative_N_m(const Vector<double> &N_m,
                 const Vector<double> &J_2,
                 const Vector<double> &T) const;

  /** Calculate the partial derivative \f$\partial \dot{N_m} / \partial N_m\f$
   */
  double
  derivative2_N_m_N_m(const double N_m, const double J_2, const double T) const;

  /** Calculate the creep strain rate \f$\dot{\varepsilon^c_{ij}} =
   * \frac{b v N_m}{2 F \sqrt{J_2}} S_{ij}\f$
   */
  double
  derivative_strain(const double N_m,
                    const double J_2,
                    const double T,
                    const double S) const;

  /** Same as above, for \c vector arguments
   */
  std::vector<double>
  derivative_strain(const std::vector<double> &N_m,
                    const std::vector<double> &J_2,
                    const std::vector<double> &T,
                    const std::vector<double> &S) const;

  /** Same as above, for \c Vector arguments
   */
  Vector<double>
  derivative_strain(const Vector<double> &N_m,
                    const Vector<double> &J_2,
                    const Vector<double> &T,
                    const Vector<double> &S) const;

  /** Calculate the dislocation velocity \f$v =
   * k_0 \tau_\mathrm{eff}^p \exp\left(-\frac{Q}{k_B T}\right)\f$
   */
  double
  dislocation_velocity(const double N_m,
                       const double J_2,
                       const double T) const;

  /** Same as above, for \c vector arguments
   */
  std::vector<double>
  dislocation_velocity(const std::vector<double> &N_m,
                       const std::vector<double> &J_2,
                       const std::vector<double> &T) const;

  /** Same as above, for \c Vector arguments
   */
  Vector<double>
  dislocation_velocity(const Vector<double> &N_m,
                       const Vector<double> &J_2,
                       const Vector<double> &T) const;

  /** Time integration using the forward Euler method (explicit)
   */
  void
  integrate_Euler();

  /** Time integration using the midpoint method (explicit).
   * Also known as RK2
   */
  void
  integrate_midpoint();

  /** Time integration using analytical expression for linearized \f$N_m\f$
   */
  void
  integrate_linearized_N_m();

  /** Time integration using analytical expression for linearized \f$N_m\f$ but
   *  the midpoint method for creep strain
   */
  void
  integrate_linearized_N_m_midpoint();

  /** Time integration using the backward Euler method (implicit)
   */
  void
  integrate_implicit();

  /** Helper method for creating output file name
   *
   * @returns \c "-<dim>d-order<order>-t<time>"
   */
  std::string
  output_name_suffix() const;

  /** Stress solver.
   *  To avoid redundancy, mesh, finite element, degree handler and temperature
   *  field is not stored in \c DislocationSolver but in \c StressSolver
   */
  StressSolver<dim> stress_solver;

  /** Dislocation density \f$N_m\f$, m<sup>-2</sup>
   */
  Vector<double> dislocation_density;

  /** Locations of probe points
   */
  std::vector<Point<dim>> probes;

  /** Parameter handler
   */
  ParameterHandler prm;

  /** Time integration scheme.
   * Default: Euler
   */
  std::string time_scheme;

  /** Time stepping: current time \f$t\f$, s
   */
  double current_time;

  /** Time stepping: current time step \f$\Delta t\f$, s
   */
  double current_time_step;

  /** Peierls potential \f$Q\f$ (temperature function), eV
   */
  FunctionParser<1> m_Q;

  /** Strain hardening factor \f$D\f$ (temperature function), N m<sup>-1</sup>
   */
  FunctionParser<1> m_D;

  /** Magnitude of Burgers vector \f$b\f$, m
   */
  double m_b;

  /** Material constant \f$K\f$, m N<sup>-1</sup>
   */
  double m_K;

  /** Material constant \f$k_0, \text{m}^{2p+l}\;\text{N}^{-p}\;\text{s}^{-1}\f$
   */
  double m_k_0;

  /** Material constant \f$l\f$, dimensionless
   */
  double m_l;

  /** Material constant \f$p\f$, dimensionless
   */
  double m_p;

  /** Average Schmid factor \f$S\f$, dimensionless
   */
  double m_S;

  /** Average Taylor factor \f$F\f$, dimensionless
   */
  double m_F;

  /** Boltzmann constant \f$k_B\f$, eV/K
   */
  static constexpr double m_k_B = 8.617e-5;
};


// IMPLEMENTATION

template <int dim>
DislocationSolver<dim>::DislocationSolver(const unsigned int order)
  : stress_solver(order)
  , current_time(0)
  , current_time_step(0)
{
  std::cout << "Creating dislocation density solver, order=" << order
            << ", dim=" << dim
            << " ("
#ifdef DEBUG
               "Debug"
#else
               "Release"
#endif
               ")\n";

  const std::string info_T = " (accepts temperature function)";

  // Physical parameters from https://doi.org/10.1016/j.jcrysgro.2016.05.027
  prm.declare_entry("Burgers vector",
                    "3.8e-10",
                    Patterns::Double(0),
                    "Magnitude of Burgers vector in m");

  prm.declare_entry("Peierls potential",
                    "2.17",
                    Patterns::Anything(),
                    "Peierls potential in eV" + info_T);

  prm.declare_entry("Strain hardening factor",
                    "4.3",
                    Patterns::Anything(),
                    "Strain hardening factor in N/m" + info_T);

  prm.declare_entry("Material constant K",
                    "3.1e-4",
                    Patterns::Double(0),
                    "Material constant K in m/N");

  prm.declare_entry("Material constant k_0",
                    "8.6e-4",
                    Patterns::Double(0),
                    "Material constant k_0 in m^(2p+l)/N^p/s");

  prm.declare_entry("Material constant l",
                    "1.0",
                    Patterns::Double(0),
                    "Material constant l (dimensionless)");

  prm.declare_entry("Material constant p",
                    "1.1",
                    Patterns::Double(0),
                    "Material constant p (dimensionless)");

  prm.declare_entry("Average Schmid factor",
                    "1.0",
                    Patterns::Double(0),
                    "Average Schmid factor S (dimensionless)");

  prm.declare_entry("Average Taylor factor",
                    "1.0",
                    Patterns::Double(0),
                    "Average Taylor factor F (dimensionless)");


  prm.declare_entry("Initial dislocation density",
                    "1e6",
                    Patterns::Double(0),
                    "Initial dislocation density in m^-2");


  prm.declare_entry("Time step",
                    "1",
                    Patterns::Double(0),
                    "Time step in seconds");

  prm.declare_entry("Min time step",
                    "0",
                    Patterns::Double(0),
                    "Minimum time step in seconds (optional, 0 - disabled)");

  prm.declare_entry("Max time step",
                    "0",
                    Patterns::Double(0),
                    "Maximum time step in seconds (optional, 0 - disabled)");

  prm.declare_entry(
    "Max relative time step increase",
    "0",
    Patterns::Double(0),
    "Maximum relative time step increase (optional, 0 - disabled)");

  prm.declare_entry("Max v*dt",
                    "0",
                    Patterns::Double(0),
                    "Maximum v*dt for adaptive time-stepping (0 - disabled)");

  prm.declare_entry(
    "Max dstrain_c",
    "0",
    Patterns::Double(0),
    "Maximum creep strain change for adaptive time-stepping (0 - disabled)");

  prm.declare_entry(
    "Max relative dN_m",
    "0",
    Patterns::Double(0),
    "Maximum relative dislocation density change for adaptive time-stepping (0 - disabled)");

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
DislocationSolver<dim>::solve(const double stress_only)
{
  if (stress_only)
    {
      // calculate stresses and nothing more
      stress_solver.solve();
      update_time_step();
      output_probes();
      return true;
    }

  advance_time();
  const double dt    = get_time_step();
  const double t     = get_time();
  const double t_max = get_max_time();

  std::cout << "Time " << t << " s\n";


  if (time_scheme == "Euler")
    integrate_Euler();
  else if (time_scheme == "Midpoint" || time_scheme == "RK2")
    integrate_midpoint();
  else if (time_scheme == "Linearized N_m")
    integrate_linearized_N_m();
  else if (time_scheme == "Linearized N_m midpoint" ||
           time_scheme == "Linearized N_m RK2")
    integrate_linearized_N_m_midpoint();
  else if (Utilities::match_at_string_start(time_scheme, "Implicit"))
    integrate_implicit();
  else
    AssertThrow(false, ExcNotImplemented());

  output_probes();

  if (dt > 0 && t + 1e-4 * dt >= t_max)
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
DislocationSolver<dim>::get_displacement() const
{
  return stress_solver.get_displacement();
}

template <int dim>
BlockVector<double> &
DislocationSolver<dim>::get_displacement()
{
  return stress_solver.get_displacement();
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
  return current_time;
}

template <int dim>
double &
DislocationSolver<dim>::get_time()
{
  return current_time;
}

template <int dim>
double
DislocationSolver<dim>::get_time_step() const
{
  return current_time_step;
}

template <int dim>
double &
DislocationSolver<dim>::get_time_step()
{
  return current_time_step;
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
  dislocation_density.add(prm.get_double("Initial dislocation density"));
}

template <int dim>
void
DislocationSolver<dim>::add_probe(const Point<dim> &p)
{
  probes.push_back(p);
}

template <int dim>
void
DislocationSolver<dim>::load_data()
{
  Timer timer;

  const std::string s = output_name_suffix();

  read_data(get_temperature(), "temperature" + s);
  read_data(get_dislocation_density(), "dislocation_density" + s);
  read_data(get_displacement(), "displacement" + s);
  read_data(get_strain_c(), "strain_c" + s);
  // skip calculated quantities (stresses)

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
DislocationSolver<dim>::output_data() const
{
  Timer timer;

  const std::string s = output_name_suffix();

  write_data(get_temperature(), "temperature" + s);
  write_data(get_dislocation_density(), "dislocation_density" + s);
  write_data(get_displacement(), "displacement" + s);
  write_data(get_stress(), "stress" + s);
  write_data(get_stress_deviator(), "stress_deviator" + s);
  write_data(get_stress_J_2(), "stress_J_2" + s);
  write_data(get_strain_c(), "strain_c" + s);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
DislocationSolver<dim>::output_vtk() const
{
  Timer timer;

  const DoFHandler<dim> &   dh = get_dof_handler();
  const FiniteElement<dim> &fe = dh.get_fe();

  const std::string file_name =
    "result-dislocation" + output_name_suffix() + ".vtk";
  std::cout << "Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh);

  const Vector<double> &T   = get_temperature();
  const Vector<double> &N_m = get_dislocation_density();

  data_out.add_data_vector(T, "T");
  data_out.add_data_vector(N_m, "N_m");

  Vector<double> Q(T.size());
  for (unsigned int i = 0; i < T.size(); ++i)
    Q[i] = calc_Q(T[i]);
  data_out.add_data_vector(Q, "Q");

  Vector<double> D(T.size());
  for (unsigned int i = 0; i < T.size(); ++i)
    D[i] = calc_D(T[i]);
  data_out.add_data_vector(D, "D");

  const BlockVector<double> &displacement = get_displacement();
  for (unsigned int i = 0; i < displacement.n_blocks(); ++i)
    {
      const std::string name = "displacement_" + std::to_string(i);
      data_out.add_data_vector(displacement.block(i), name);
    }

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

  const BlockVector<double> &epsilon_c = get_strain_c();
  for (unsigned int i = 0; i < epsilon_c.n_blocks(); ++i)
    {
      const std::string name = "epsilon_c_" + std::to_string(i);
      data_out.add_data_vector(epsilon_c.block(i), name);
    }

  const Vector<double> &stress_J_2 = get_stress_J_2();

  const Vector<double> tau     = tau_eff(N_m, stress_J_2, T);
  const Vector<double> dot_N_m = derivative_N_m(N_m, stress_J_2, T);
  const Vector<double> v       = dislocation_velocity(N_m, stress_J_2, T);

  BlockVector<double> dot_epsilon_c(StressSolver<dim>::n_components);
  for (unsigned int i = 0; i < dot_epsilon_c.n_blocks(); ++i)
    {
      dot_epsilon_c.block(i) =
        derivative_strain(N_m, stress_J_2, T, stress_deviator.block(i));
    }

  data_out.add_data_vector(stress_J_2, "stress_J_2");
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

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
DislocationSolver<dim>::output_mesh() const
{
  Timer timer;

  const std::string file_name = "mesh" + output_name_suffix() + ".msh";
  std::cout << "Saving to '" << file_name << "'";

  std::ofstream output(file_name);
  output << std::setprecision(16);

  GridOut grid_out;
  grid_out.set_flags(GridOutFlags::Msh(true));
  grid_out.write_msh(get_mesh(), output);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
DislocationSolver<dim>::initialize_parameters()
{
  std::cout << "Intializing parameters";

  const std::string m_Q_expression = prm.get("Peierls potential");
  m_Q.initialize("T", m_Q_expression, typename FunctionParser<1>::ConstMap());

  const std::string m_D_expression = prm.get("Strain hardening factor");
  m_D.initialize("T", m_D_expression, typename FunctionParser<1>::ConstMap());

  m_b   = prm.get_double("Burgers vector");
  m_K   = prm.get_double("Material constant K");
  m_k_0 = prm.get_double("Material constant k_0");
  m_l   = prm.get_double("Material constant l");
  m_p   = prm.get_double("Material constant p");
  m_S   = prm.get_double("Average Schmid factor");
  m_F   = prm.get_double("Average Taylor factor");

  time_scheme = prm.get("Time scheme");

  get_time_step() = prm.get_double("Time step");

  std::cout << "  done\n";

  std::cout << "b=" << m_b << "\n"
            << "Q=" << m_Q_expression << "\n"
            << "D=" << m_D_expression << "\n"
            << "K=" << m_K << "\n"
            << "k_0=" << m_k_0 << "\n"
            << "l=" << m_l << "\n"
            << "p=" << m_p << "\n"
            << "S=" << m_S << "\n"
            << "F=" << m_F << "\n"
            << "k_B=" << m_k_B << "\n"
            << "time_scheme=" << time_scheme << "\n";
}

template <int dim>
double
DislocationSolver<dim>::get_time_step_min() const
{
  return prm.get_double("Min time step");
}

template <int dim>
double
DislocationSolver<dim>::get_time_step_max() const
{
  return prm.get_double("Max time step");
}

template <int dim>
void
DislocationSolver<dim>::limit_time_step()
{
  double &dt = get_time_step();

  const double dt_min = get_time_step_min();
  if (dt_min > 0 && dt < dt_min)
    dt = dt_min;

  const double dt_max = get_time_step_max();
  if (dt_max > 0 && dt > dt_max)
    dt = dt_max;
}

template <int dim>
void
DislocationSolver<dim>::update_time_step()
{
  double &     dt            = get_time_step();
  const double dt_prev       = dt;
  const double v_dt_max      = prm.get_double("Max v*dt");
  const double dstrain_c_max = prm.get_double("Max dstrain_c");
  const double dN_m_rel_max  = prm.get_double("Max relative dN_m");
  const double dt_rel_max = prm.get_double("Max relative time step increase");

  const Vector<double> &     T   = get_temperature();
  const Vector<double> &     N_m = get_dislocation_density();
  const Vector<double> &     J_2 = get_stress_J_2();
  const BlockVector<double> &S   = get_stress_deviator();

  std::vector<double> time_steps;

  if (v_dt_max > 0)
    {
      const Vector<double> v = dislocation_velocity(N_m, J_2, T);

      const double v_max = v.linfty_norm();

      if (v_max > 0)
        {
          const double dt = v_dt_max / v_max;
          time_steps.push_back(dt);
          std::cout << "dt=" << dt << " s, "
                    << "v_max=" << v_max << " m/s\n";
        }
    }

  if (dstrain_c_max > 0)
    {
      for (unsigned int i = 0; i < S.n_blocks(); ++i)
        {
          const Vector<double> dot_e_c =
            derivative_strain(N_m, J_2, T, S.block(i));

          const double dot_strain_c = dot_e_c.linfty_norm();

          if (dot_strain_c > 0)
            {
              const double dt = dstrain_c_max / dot_strain_c;
              time_steps.push_back(dt);
              std::cout << "dt=" << dt << " s, "
                        << "dot_strain_c_" << i << "_max=" << dot_strain_c
                        << " 1/s\n";
            }
        }
    }

  if (dN_m_rel_max > 0)
    {
      Vector<double> dn = derivative_N_m(N_m, J_2, T);
      for (unsigned int i = 0; i < dn.size(); ++i)
        {
          if (N_m[i] > 0)
            dn[i] /= N_m[i];
          else
            dn[i] = 0;
        }

      const double dot_N_m_rel = dn.linfty_norm();

      if (dot_N_m_rel > 0)
        {
          const double dt = dN_m_rel_max / dot_N_m_rel;
          time_steps.push_back(dt);
          std::cout << "dt=" << dt << " s, "
                    << "dot_N_m_rel_max=" << dot_N_m_rel << "\n";
        }
    }

  if (!time_steps.empty())
    dt = *std::min_element(time_steps.begin(), time_steps.end());

  if (dt_rel_max > 0)
    {
      const double dt_change = dt - dt_prev;

      // limit the time step increase
      if (dt_change > dt_prev * dt_rel_max)
        dt = dt_prev + dt_prev * dt_rel_max;
    }

  limit_time_step();

  // stop exactly at max time
  if (get_time() + (1 + 1e-4) * dt >= get_max_time())
    dt = get_max_time() - get_time();
}

template <int dim>
void
DislocationSolver<dim>::advance_time()
{
  update_time_step();
  get_time() += get_time_step();
}

template <int dim>
void
DislocationSolver<dim>::output_probes() const
{
  Timer timer;

  std::stringstream ss;
  ss << "probes-dislocation-" << dim << "d.txt";
  const std::string file_name = ss.str();

  std::cout << "Saving values at probe points to '" << file_name << "'";

  const unsigned int N = probes.size();

  const double t  = get_time();
  const double dt = get_time_step();

  const BlockVector<double> &s   = get_stress();
  const BlockVector<double> &S   = get_stress_deviator();
  const BlockVector<double> &e_c = get_strain_c();

  if (t == 0)
    {
      // write header at the first time step
      std::ofstream output(file_name);

      for (unsigned int i = 0; i < N; ++i)
        output << "# probe " << i << ":\t" << probes[i] << "\n";

      output << "t[s]"
             << "\tdt[s]"
             << "\tT_min[K]"
             << "\tT_max[K]"
             << "\tN_m_min[m^-2]"
             << "\tN_m_max[m^-2]"
             << "\tdot_N_m_min[m^-2s^-1]"
             << "\tdot_N_m_max[m^-2s^-1]"
             << "\tv_min[ms^-1]"
             << "\tv_max[ms^-1]"
             << "\tstress_min[Pa]"
             << "\tstress_max[Pa]"
             << "\tstrain_c_min[-]"
             << "\tstrain_c_max[-]"
             << "\tdot_strain_c_min[s^-1]"
             << "\tdot_strain_c_max[s^-1]"
             << "\ttau_eff_min[Pa]"
             << "\ttau_eff_max[Pa]"
             << "\tJ_2_min[Pa^2]"
             << "\tJ_2_max[Pa^2]";

      for (unsigned int i = 0; i < N; ++i)
        {
          output << "\tT_" << i << "[K]"
                 << "\tN_m_" << i << "[m^-2]"
                 << "\tdot_N_m_" << i << "[m^-2s^-1]"
                 << "\tv_" << i << "[ms^-1]";

          for (unsigned int j = 0; j < s.n_blocks(); ++j)
            output << "\tstress_" << j << "_" << i << "[Pa]";

          for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
            output << "\tstrain_c_" << j << "_" << i << "[-]";

          for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
            output << "\tdot_strain_c_" << j << "_" << i << "[s^-1]";

          output << "\ttau_eff_" << i << "[Pa]"
                 << "\tJ_2_" << i << "[Pa^2]";
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

  std::vector<std::vector<double>> values_S(S.n_blocks());
  for (unsigned int i = 0; i < S.n_blocks(); ++i)
    values_S[i] = get_field_at_probes(S.block(i));

  std::vector<std::vector<double>> values_e_c(e_c.n_blocks());
  for (unsigned int i = 0; i < e_c.n_blocks(); ++i)
    values_e_c[i] = get_field_at_probes(e_c.block(i));

  // calculate additional fields
  const std::vector<double> values_tau =
    tau_eff(values_N_m, values_J_2, values_T);

  const std::vector<double> values_dot_N_m =
    derivative_N_m(values_N_m, values_J_2, values_T);

  const std::vector<double> values_v =
    dislocation_velocity(values_N_m, values_J_2, values_T);

  std::vector<std::vector<double>> values_dot_e_c(e_c.n_blocks());
  for (unsigned int i = 0; i < e_c.n_blocks(); ++i)
    values_dot_e_c[i] =
      derivative_strain(values_N_m, values_J_2, values_T, values_S[i]);

  BlockVector<double> dot_e_c(e_c.n_blocks());
  for (unsigned int i = 0; i < e_c.n_blocks(); ++i)
    dot_e_c.block(i) = derivative_strain(N_m, J_2, T, S.block(i));

  // header is already written, append values at the current time step
  std::ofstream output(file_name, std::ios::app);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  const auto limits_T       = minmax(T);
  const auto limits_N_m     = minmax(N_m);
  const auto limits_dot_N_m = minmax(derivative_N_m(N_m, J_2, T));
  const auto limits_v       = minmax(dislocation_velocity(N_m, J_2, T));
  const auto limits_s       = minmax(s);
  const auto limits_e_c     = minmax(e_c);
  const auto limits_dot_e_c = minmax(dot_e_c);
  const auto limits_tau     = minmax(tau_eff(N_m, J_2, T));
  const auto limits_J_2     = minmax(J_2);

  output << t << '\t' << dt;
  output << '\t' << limits_T.first << '\t' << limits_T.second;
  output << '\t' << limits_N_m.first << '\t' << limits_N_m.second;
  output << '\t' << limits_dot_N_m.first << '\t' << limits_dot_N_m.second;
  output << '\t' << limits_v.first << '\t' << limits_v.second;
  output << '\t' << limits_s.first << '\t' << limits_s.second;
  output << '\t' << limits_e_c.first << '\t' << limits_e_c.second;
  output << '\t' << limits_dot_e_c.first << '\t' << limits_dot_e_c.second;
  output << '\t' << limits_tau.first << '\t' << limits_tau.second;
  output << '\t' << limits_J_2.first << '\t' << limits_J_2.second;

  for (unsigned int i = 0; i < N; ++i)
    {
      output << '\t' << values_T[i] << '\t' << values_N_m[i] << '\t'
             << values_dot_N_m[i] << '\t' << values_v[i];

      for (unsigned int j = 0; j < s.n_blocks(); ++j)
        output << '\t' << values_s[j][i];

      for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
        output << '\t' << values_e_c[j][i];

      for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
        output << '\t' << values_dot_e_c[j][i];

      output << '\t' << values_tau[i] << '\t' << values_J_2[i];
    }
  output << "\n";

  std::cout << " " << format_time(timer) << "\n";
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
DislocationSolver<dim>::calc_Q(const double T) const
{
  return m_Q.value(Point<1>(T));
}

template <int dim>
double
DislocationSolver<dim>::calc_D(const double T) const
{
  return m_D.value(Point<1>(T));
}

template <int dim>
double
DislocationSolver<dim>::tau_eff(const double N_m,
                                const double J_2,
                                const double T) const
{
  return std::max(m_S * std::sqrt(J_2) - calc_D(T) * std::sqrt(N_m), 0.0);
}

template <int dim>
std::vector<double>
DislocationSolver<dim>::tau_eff(const std::vector<double> &N_m,
                                const std::vector<double> &J_2,
                                const std::vector<double> &T) const
{
  const unsigned int N = N_m.size();

  std::vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = tau_eff(N_m[i], J_2[i], T[i]);

  return result;
}

template <int dim>
Vector<double>
DislocationSolver<dim>::tau_eff(const Vector<double> &N_m,
                                const Vector<double> &J_2,
                                const Vector<double> &T) const
{
  const unsigned int N = N_m.size();

  Vector<double> result(N);

  for (unsigned int i = 0; i < N; ++i)
    result[i] = tau_eff(N_m[i], J_2[i], T[i]);

  return result;
}

template <int dim>
double
DislocationSolver<dim>::derivative_N_m(const double N_m,
                                       const double J_2,
                                       const double T) const
{
  const double tau = tau_eff(N_m, J_2, T);

  return m_K * m_k_0 * std::pow(tau, m_p + m_l) *
         std::exp(-calc_Q(T) / (m_k_B * T)) * N_m;
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
  const double tau = tau_eff(N_m, J_2, T);

  return m_K * m_k_0 * std::exp(-calc_Q(T) / (m_k_B * T)) *
         (std::pow(tau, m_p + m_l) - (m_p + m_l) *
                                       std::pow(tau, m_p + m_l - 1) *
                                       std::sqrt(N_m) * calc_D(T) / 2);
}

template <int dim>
double
DislocationSolver<dim>::derivative_strain(const double N_m,
                                          const double J_2,
                                          const double T,
                                          const double S) const
{
  if (J_2 == 0)
    return 0;

  const double tau = tau_eff(N_m, J_2, T);

  return m_b * m_k_0 * N_m * std::pow(tau, m_p) *
         std::exp(-calc_Q(T) / (m_k_B * T)) * S / (2 * m_F * std::sqrt(J_2));
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
  const double tau = tau_eff(N_m, J_2, T);

  return m_k_0 * std::pow(tau, m_p) * std::exp(-calc_Q(T) / (m_k_B * T));
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
      const double d_N_m = dx_analytical(a, b, dt);

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
      N_m[i] += dx_analytical(a, b, dt / 2);
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
      N_m[i] = N_m_0[i] + dx_analytical(a, b, dt);
    }

  stress_solver.solve();
}

template <int dim>
void
DislocationSolver<dim>::integrate_implicit()
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

  // get number of fixed-point iterations
  const std::vector<std::string> tmp =
    Utilities::split_string_list(time_scheme, ' ');
  const unsigned int n_iterations = tmp.size() > 1 ? std::stoul(tmp.back()) : 1;

  for (unsigned int k = 0; k <= n_iterations; ++k)
    {
      // k=0: initial approximation (forward Euler)

      std::cout << "Fixed point iteration " << k << " of " << n_iterations
                << "\n";

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
}

template <int dim>
std::string
DislocationSolver<dim>::output_name_suffix() const
{
  std::stringstream ss;
  ss << "-" << dim << "d-order" << get_dof_handler().get_fe().degree << "-t"
     << get_time();
  return ss.str();
}

#endif