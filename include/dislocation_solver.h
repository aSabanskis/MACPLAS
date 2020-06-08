#ifndef macplas_dislocation_solver_h
#define macplas_dislocation_solver_h

#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/// Class for calculation of the time-dependent dislocation density
template <int dim>
class DislocationSolver
{
public:
  /// Constructor
  DislocationSolver();

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
DislocationSolver<dim>::DislocationSolver()
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