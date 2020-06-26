#ifndef macplas_dislocation_solver_h
#define macplas_dislocation_solver_h

#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

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
  DislocationSolver(const unsigned int order = 2);

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

  /// Get degrees of freedom for temperature
  const DoFHandler<dim> &
  get_dof_handler() const;

  /// Get stress solver (for setting boundary conditions)
  StressSolver<dim> &
  get_stress_solver();

  /// Initialize fields
  void
  initialize();

  /// Save results to disk
  void
  output_results() const;

  /// Save mesh to disk
  void
  output_mesh() const;

private:
  /// Initialize the solver parameters from \c dislocation.prm.

  /// If it doesn't exist, the default parameter values are written to
  /// \c dislocation-default.prm.
  void
  initialize_parameters();

  /// Calculate the effective stress \f$\tau_\mathrm{eff}\f$
  double
  tau_eff(const double N_m, const double J_2) const;

  /// Calculate the time derivative of dislocation density \f$\dot{N_m}\f$
  double
  derivative_N_m(const double N_m, const double J_2, const double T) const;

  /// Calculate the creep strain rate \f$\dot{\varepsilon^c_{ij}}\f$
  double
  derivative_strain(const double N_m,
                    const double J_2,
                    const double T,
                    const double S) const;

  /// Stress solver

  /// To avoid redundancy, mesh, finite element, degree handler and temperature
  /// field is not stored in DislocationSolver but in StressSolver
  StressSolver<dim> stress_solver;

  Vector<double> dislocation_density; ///< Dislocation density, m<sup>-2</sup>

  /// Creep strain
  std::vector<Tensor<1, StressSolver<dim>::n_components>> epsilon_c;

  ParameterHandler prm; ///< Parameter handler

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
void
DislocationSolver<dim>::initialize()
{
  stress_solver.initialize(); // prints info

  const unsigned int n_dofs_temp = get_temperature().size();

  dislocation_density.reinit(n_dofs_temp);
  epsilon_c.resize(n_dofs_temp);
  std::fill(epsilon_c.begin(),
            epsilon_c.end(),
            Tensor<1, StressSolver<dim>::n_components>());
}

template <int dim>
void
DislocationSolver<dim>::output_results() const
{
  Timer timer;

  const DoFHandler<dim> &   dh = get_dof_handler();
  const FiniteElement<dim> &fe = dh.get_fe();

  std::stringstream ss;
  ss << "result-" << dim << "d-order" << fe.degree << ".vtk";
  const std::string file_name = ss.str();
  std::cout << "Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh);

  data_out.add_data_vector(get_temperature(), "T");
  data_out.add_data_vector(get_dislocation_density(), "dislocation_density");

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

  std::cout << "  done\n";

  std::cout << "b=" << m_b << "\n"
            << "Q=" << m_Q << "\n"
            << "D=" << m_D << "\n"
            << "K=" << m_K << "\n"
            << "k_0=" << m_k_0 << "\n"
            << "l=" << m_l << "\n"
            << "p=" << m_p << "\n"
            << "k_B=" << m_k_B << "\n";
}

template <int dim>
double
DislocationSolver<dim>::tau_eff(const double N_m, const double J_2) const
{
  return std::max(std::sqrt(J_2) - m_D * std::sqrt(N_m), 0.0);
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

#endif