#ifndef macplas_temperature_solver_h
#define macplas_temperature_solver_h

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "utilities.h"

using namespace dealii;

#if DEAL_II_VERSION_GTE(9, 5, 0)
template <int dim, typename RangeNumberType = double>
using ConstantFunction = Functions::ConstantFunction<dim, RangeNumberType>;

template <int dim, typename RangeNumberType = double>
using ZeroFunction = Functions::ZeroFunction<dim, RangeNumberType>;
#endif

/** Stefan–Boltzmann constant \f$\sigma_\mathrm{SB}\f$,
 * W m<sup>-2</sup> K<sup>-4</sup>
 */
static const double sigma_SB = 5.67e-8;


/** Data structure for thermal radiation and incoming heat flux density BC
 * \f$q = \sigma_\mathrm{SB} \varepsilon(T) (T^4 - T_\mathrm{amb}^4) -
 * q_\mathrm{in}\f$.
 * The temperature dependence of \f$\varepsilon\f$ is
 * taken into account.
 */
struct radiation_heat_flux_data
{
  /** Incoming heat flux density \f$q_\mathrm{in}\f$, W m<sup>-2</sup>
   */
  Vector<double> q_in;

  /** Temperature-dependent emissivity \f$\varepsilon(T)\f$, dimensionless
   */
  std::function<double(const double)> emissivity;

  /** Emissivity temperature derivative \f$d\varepsilon(T)/dT\f$, K<sup>-1</sup>
   */
  std::function<double(const double)> emissivity_deriv;

  /** Ambient temperature \f$T_\mathrm{amb}\f$, K
   */
  double T_amb;
};

/** Data structure for convective cooling BC (Newton's law of cooling)
 * \f$q = h (T - T_\mathrm{ref})\f$
 */
struct convective_cooling_data
{
  /** Heat transfer coefficient \f$h\f$, W m<sup>-2</sup> K<sup>-1</sup>
   */
  double h;

  /** Reference temperature \f$T_\mathrm{ref}\f$, K
   */
  double T_ref;
};


/** Class for calculation of the time-dependent temperature field
 */
template <int dim>
class TemperatureSolver
{
public:
  /** Constructor.
   * Initializes the solver parameters from \c temperature.prm.
   * If it doesn't exist, the default parameter values are written to
   * \c temperature-default.prm.
   * Default values are used and written to \c temperature.prm if
   * \c use_default_prm parameter is specified.
   *
   * \todo Redesign boundary condition handling to keep TemperatureSolver
   * maintainable in the future
   * \todo Automatically calculate temperature derivatives needed for the
   * Newton's method
   */
  explicit TemperatureSolver(const unsigned int order           = 2,
                             const bool         use_default_prm = false);

  /** Solver name
   */
  std::string
  solver_name() const;

  /** Calculate the temperature field.
   * @returns \c true if the final time has not been reached
   */
  bool
  solve(const bool skip_time_advance = false);

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

  /** Get volumetric heat source \f$\dot{q}\f$, W m<sup>-3</sup>
   */
  const Vector<double> &
  get_heat_source() const;

  /** Get volumetric heat source \f$\dot{q}\f$, W m<sup>-3</sup>
   */
  Vector<double> &
  get_heat_source();

  /** Current time \f$t\f$, s
   */
  double
  get_time() const;

  /** Current time \f$t\f$, s
   */
  double &
  get_time();

  /** Time step \f$\Delta t\f$, s
   */
  double
  get_time_step() const;

  /** Time step \f$\Delta t\f$, s
   */
  double &
  get_time_step();

  /** Final time, s
   */
  double
  get_max_time() const;

  /** Get parameters TemperatureSolver::prm
   */
  const ParameterHandler &
  get_parameters() const;

  /** Get parameters TemperatureSolver::prm
   */
  ParameterHandler &
  get_parameters();

  /** Initialize DOF handler and fields
   */
  void
  initialize();

  /** Get coordinates of boundary DOFs
   */
  void
  get_boundary_points(const unsigned int       id,
                      std::vector<Point<dim>> &points,
                      std::vector<bool> &      boundary_dofs) const;

  /** Get coordinates of DOFs
   */
  void
  get_support_points(std::vector<Point<dim>> &points) const;

  /** Extract coordinates of boundary DOFs
   */
  std::map<unsigned int, Point<dim>>
  get_boundary_dofs(const unsigned int boundary_id) const;

  /** Evaluate values of \c source field at \c points.
   * Handles error for points outside the mesh by setting the value to zero.
   */
  std::vector<double>
  get_field_at_points(const Vector<double> &         source,
                      const std::vector<Point<dim>> &points) const;

  /** Get finite element degree
   */
  unsigned int
  get_degree() const;

  /** Get degrees of freedom for temperature
   */
  const DoFHandler<dim> &
  get_dof_handler() const;

  /** Calculate the temperature-dependent thermal conductivity \f$\lambda(T)\f$,
   * W m<sup>-1</sup> K<sup>-1</sup>
   */
  double
  calc_lambda(const double T) const;

  /** Calculate the derivative of thermal conductivity \f$d\lambda(T)/dT\f$,
   * W m<sup>-1</sup> K<sup>-2</sup>
   */
  double
  calc_derivative_lambda(const double T) const;

  /** Calculate the temperature-dependent density \f$\rho\f$, kg m<sup>-3</sup>
   */
  double
  calc_rho(const double T) const;

  /** Calculate the temperature-dependent specific heat capacity \f$c_p\f$,
   * J kg K<sup>-1</sup>
   */
  double
  calc_c_p(const double T) const;

  /** Calculate the temperature-dependent product of density and specific heat
   * capacity \f$\rho c_p\f$, J m<sup>-3</sup> K<sup>-1</sup>
   */
  double
  calc_rho_c_p(const double T) const;

  /** Calculate the vertical velocity \f$V_z\f$, m s<sup>-1</sup>.
   * The reason for a separate function is that the value can be changed through
   * parameters during the simulation.
   */
  double
  calc_V_z() const;

  /** Clear all applied boundary conditions
   */
  void
  clear_bcs();

  /** Set first-type boundary condition \f$T = \mathrm{val}\f$
   */
  void
  set_bc1(const unsigned int id, const double val);

  /** Same as above, for non-homogeneous field
   */
  void
  set_bc1(const unsigned int id, const Vector<double> &val);

  /** Set thermal radiation and incoming heat flux density boundary condition
   * \f$q = \sigma_\mathrm{SB} \varepsilon(T) (T^4 - T_\mathrm{amb}^4) -
   * q_\mathrm{in}\f$, see radiation_heat_flux_data
   */
  void
  set_bc_rad_mixed(const unsigned int                  id,
                   const Vector<double> &              q_in,
                   std::function<double(const double)> emissivity,
                   std::function<double(const double)> emissivity_deriv,
                   const double                        T_amb = 0);

  /** Set convective cooling boundary condition
   * \f$q = h (T - T_\mathrm{ref})\f$, see convective_cooling_data
   */
  void
  set_bc_convective(const unsigned int id, const double h, const double T_ref);

  /** Add probe point
   */
  void
  add_probe(const Point<dim> &p);

  /** Add user-defined output value
   */
  void
  add_output(const std::string &name, const double value = 0);

  /** Add user-defined field
   */
  void
  add_field(const std::string &name, const Vector<double> &value);

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

  /** Save results at DOFs of boundary \c id to disk
   */
  void
  output_boundary_values(const unsigned int id) const;

  /** Save mesh to disk in \c msh format
   */
  void
  output_mesh() const;

  /** Calculate and write temperature-dependent parameters to disk
   */
  void
  output_parameter_table(const double       T1 = 250,
                         const double       T2 = 1700,
                         const unsigned int n  = 30) const;

private:
  /** Initialize parameters. Called by the constructor
   */
  void
  initialize_parameters();

  /** Initialize data before calculation
   */
  void
  prepare_for_solve();

  /** Apply first-type boundary condition previously set by
   * TemperatureSolver::set_bc1.
   */
  void
  apply_bc1();

  /** Assemble the system matrix and right-hand-side vector
   */
  void
  assemble_system();


  /** Structure that holds scratch data
   */
  struct AssemblyScratchData
  {
    AssemblyScratchData(const Quadrature<dim>     quadrature,
                        const Quadrature<dim - 1> face_quadrature,
                        const FiniteElement<dim> &fe);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);

    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;

    std::vector<double>         T_q;
    std::vector<double>         T_prev_q;
    std::vector<double>         dot_q_q;
    std::vector<Tensor<1, dim>> grad_T_q;
    std::vector<double>         T_face_q;
    std::vector<double>         q_in_face_q;
  };

  /** Structure that holds local contributions
   */
  struct AssemblyCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };

  /** Local assembly function
   */
  void
  local_assemble_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData &                                 scratch_data,
    AssemblyCopyData &                                    copy_data);

  /** Copy local contributions to global
   */
  void
  copy_local_to_global(const AssemblyCopyData &copy_data);


  /** Time stepping: advance time \f$t \to t + \Delta t\f$ and update previous
   * temperature field
   */
  void
  advance_time();

  /** Solve the system of linear equations
   */
  void
  solve_system();

  /** Write temperature at probe points to disk.
   * File name \c "probes-temperature-<dim>d.txt"
   */
  void
  output_probes() const;

  /** Evaluate values of \c source field at probe points
   */
  std::vector<double>
  get_field_at_probes(const Vector<double> &source) const;

  /** Helper method for creating output file name.
   * @returns \c "-<dim>d-order<order>-t<time>"
   */
  std::string
  output_name_suffix() const;

  /** Mesh
   */
  Triangulation<dim> triangulation;

  /** Finite element
   */
  FE_Q<dim> fe;

  /** Degrees of freedom
   */
  DoFHandler<dim> dh;

  /** Sparsity pattern
   */
  SparsityPattern sparsity_pattern;

  /** System matrix
   */
  SparseMatrix<double> system_matrix;

  /** Right-hand-side vector
   */
  Vector<double> system_rhs;


  /** Temperature \f$T\f$, K
   */
  Vector<double> temperature;

  /** Temperature at the previous time step \f$T_\mathrm{prev}\f$, K
   */
  Vector<double> temperature_prev;

  /** Newton update for temperature \f$\delta T\f$, K
   */
  Vector<double> temperature_update;

  /** Volumetric heat source \f$\dot{q}\f$, W m<sup>-3</sup>
   */
  Vector<double> vol_heat_source;

  /** Density (temperature function) \f$\rho\f$, kg m<sup>-3</sup>
   */
  FunctionParser<1> m_rho;

  /** Specific heat capacity (temperature function) \f$c_p\f$,
   * J kg<sup>-1</sup> K<sup>-1</sup>
   */
  FunctionParser<1> m_c_p;

  /** Thermal conductivity \f$\lambda(T)\f$, W m<sup>-1</sup> K<sup>-1</sup>
   */
  FunctionParser<1> m_lambda;

  /** Derivative of thermal conductivity \f$d\lambda(T)/dT\f$,
   * W m<sup>-1</sup> K<sup>-2</sup>
   */
  FunctionParser<1> m_derivative_lambda;


  /** Data for first-type BC
   */
  std::map<unsigned int, Vector<double>> bc1_data;

  /** Data for thermal radiation and incoming heat flux density BC
   * radiation_heat_flux_data
   */
  std::map<unsigned int, radiation_heat_flux_data> bc_rad_mixed_data;

  /** Data for convective cooling BC convective_cooling_data
   */
  std::map<unsigned int, convective_cooling_data> bc_convective_data;


  /** User-defined fields
   */
  std::map<std::string, Vector<double>> additional_fields;

  /** Locations of probe points
   */
  std::vector<Point<dim>> probes;

  /** User-defined output values
   */
  std::map<std::string, double> additional_output;

  /** Flag for writing header of the probes file
   */
  bool probes_header_written;


  /**  Parameter handler
   */
  ParameterHandler prm;


  /** Time stepping: current time \f$t\f$, s
   */
  double current_time;

  /** Time stepping: current time step \f$\Delta t\f$, s
   */
  double current_time_step;
};


// IMPLEMENTATION

template <int dim>
TemperatureSolver<dim>::TemperatureSolver(const unsigned int order,
                                          const bool         use_default_prm)
  : fe(order)
  , dh(triangulation)
  , probes_header_written(false)
  , current_time(0)
  , current_time_step(0)
{
  std::cout << "Creating temperature solver, order=" << order << ", dim=" << dim
            << " ("
#ifdef DEBUG
               "Debug"
#else
               "Release"
#endif
               ")\n";

  prm.declare_entry("Max absolute change",
                    "1e-3",
                    Patterns::Double(0),
                    "Maximum magnitute of Newton update");

  prm.declare_entry("Max Newton iterations",
                    "6",
                    Patterns::Integer(0),
                    "Maximum number of Newton iterations (0 - unlimited)");

  prm.declare_entry("Newton step length",
                    "1",
                    Patterns::Double(0, 1),
                    "Value of Newton step length");

  prm.declare_entry("Time step",
                    "1",
                    Patterns::Double(0),
                    "Time step in seconds (0 - steady-state)");

  prm.declare_entry("Max time",
                    "10",
                    Patterns::Double(0),
                    "Maximum time in seconds");

  prm.declare_entry("Linear solver type",
                    "minres",
                    Patterns::Selection("UMFPACK|" +
                                        SolverSelector<>::get_solver_names()),
                    "Name of linear solver");

  prm.declare_entry("Linear solver iterations",
                    "1000",
                    Patterns::Integer(0),
                    "Maximum number of iterations of linear solver");

  prm.declare_entry("Linear solver tolerance",
                    "1e-8",
                    Patterns::Double(0),
                    "Tolerance (maximum residual norm) of linear solver");

  prm.declare_entry("Preconditioner type",
                    "jacobi",
                    Patterns::Selection(
                      PreconditionSelector<>::get_precondition_names()),
                    "Name of preconditioner");

  prm.declare_entry("Preconditioner relaxation",
                    "1.0",
                    Patterns::Double(0),
                    "Relaxation factor of preconditioner");

  prm.declare_entry("Log convergence full",
                    "false",
                    Patterns::Bool(),
                    "Report convergence progress of linear solver");

  prm.declare_entry("Log convergence final",
                    "true",
                    Patterns::Bool(),
                    "Report final achieved convergence of linear solver");

  prm.declare_entry("Number of cell quadrature points",
                    "0",
                    Patterns::Integer(0),
                    "Number of QGauss<dim> quadrature points (0: order+1)");

  prm.declare_entry("Number of face quadrature points",
                    "0",
                    Patterns::Integer(0),
                    "Number of QGauss<dim-1> quadrature points (0: order+1)");

  prm.declare_entry("Number of threads",
                    "0",
                    Patterns::Integer(0),
                    "Maximum number of threads to be used (0 - autodetect)");

  prm.declare_entry(
    "Output precision",
    "8",
    Patterns::Integer(1),
    "Precision of double variables for output of field and probe data");

  prm.declare_entry("Output subdivisions",
                    "0",
                    Patterns::Integer(0),
                    "Number of cell subdivisions for vtk output (0: order)");


  const std::string info_T = " (temperature function)";

  // Physical parameters from https://doi.org/10.1016/S0022-0248(03)01253-3
  prm.declare_entry("Density",
                    "2329",
                    Patterns::Anything(),
                    "Density rho in kg/m^3" + info_T);

  prm.declare_entry("Specific heat capacity",
                    "1000",
                    Patterns::Anything(),
                    "Specific heat capacity c_p in J/kg/K" + info_T);

  prm.declare_entry("Thermal conductivity",
                    "98.89 -9.41813870776526E-02*T +2.88183040644504E-05*T^2",
                    Patterns::Anything(),
                    "Thermal conductivity lambda in W/m/K" + info_T);

  prm.declare_entry("Thermal conductivity derivative",
                    "-9.41813870776526E-02 +2*2.88183040644504E-05*T",
                    Patterns::Anything(),
                    "Derivative of thermal conductivity in W/m/K^2" + info_T);

  prm.declare_entry("Velocity",
                    "0",
                    Patterns::Double(),
                    "Vertical velocity in m/s");

  if (use_default_prm)
    {
      std::ofstream of("temperature.prm");
      prm.print_parameters(of, ParameterHandler::Text);
    }
  else
    try
      {
        prm.parse_input("temperature.prm");
      }
    catch (std::exception &e)
      {
        std::cout << e.what() << "\n";

        std::ofstream of("temperature-default.prm");
        prm.print_parameters(of, ParameterHandler::Text);
      }

  initialize_parameters();
}

template <int dim>
void
TemperatureSolver<dim>::initialize_parameters()
{
  std::cout << solver_name() << "  Initializing parameters";

  const std::string m_rho_expression = prm.get("Density");
  m_rho.initialize("T",
                   m_rho_expression,
                   typename FunctionParser<1>::ConstMap());

  const std::string m_c_p_expression = prm.get("Specific heat capacity");
  m_c_p.initialize("T",
                   m_c_p_expression,
                   typename FunctionParser<1>::ConstMap());

  const std::string m_lambda_expression = prm.get("Thermal conductivity");
  m_lambda.initialize("T",
                      m_lambda_expression,
                      typename FunctionParser<1>::ConstMap());

  const std::string m_derivative_lambda_expression =
    prm.get("Thermal conductivity derivative");
  m_derivative_lambda.initialize("T",
                                 m_derivative_lambda_expression,
                                 typename FunctionParser<1>::ConstMap());


  const auto n_threads = prm.get_integer("Number of threads");
  MultithreadInfo::set_thread_limit(n_threads > 0 ? n_threads :
                                                    MultithreadInfo::n_cores());

  get_time_step() = prm.get_double("Time step");

  add_output("nNewton");
  add_output("Velocity[m/s]");

  const long int n_q_default = get_degree() + 1;

  if (prm.get_integer("Number of cell quadrature points") == 0)
    prm.set("Number of cell quadrature points", n_q_default);

  if (prm.get_integer("Number of face quadrature points") == 0)
    prm.set("Number of face quadrature points", n_q_default);

  const long int n_vtk_default = get_degree();

  if (prm.get_integer("Output subdivisions") == 0)
    prm.set("Output subdivisions", n_vtk_default);

  std::cout << "  done\n";

  std::cout << "rho=" << m_rho_expression << "\n"
            << "c_p=" << m_c_p_expression << "\n"
            << "lambda=" << m_lambda_expression << "\n"
            << "derivative_lambda=" << m_derivative_lambda_expression << "\n"
            << "V_z=" << calc_V_z() << "\n";

  std::cout << "n_q_cell=" << prm.get("Number of cell quadrature points")
            << "\n"
            << "n_q_face=" << prm.get("Number of face quadrature points")
            << "\n";

  std::cout << "n_cores=" << MultithreadInfo::n_cores() << "\n"
            << "n_threads=" << MultithreadInfo::n_threads() << "\n";
}

template <int dim>
std::string
TemperatureSolver<dim>::solver_name() const
{
  return "MACPLAS:Temperature";
}

template <int dim>
bool
TemperatureSolver<dim>::solve(const bool skip_time_advance)
{
  if (!probes_header_written)
    {
      output_probes();
      // set here to keep const-ness of output_probes
      probes_header_written = true;
    }

  // needed so that solve() could be called multiple times at the same t
  if (!skip_time_advance)
    advance_time();

  const double t     = get_time();
  const double dt    = get_time_step();
  const double t_max = get_max_time();

  const double V_z = calc_V_z();
  add_output("Velocity[m/s]", V_z);
#ifdef DEBUG
  if (dt > 0 && V_z != 0)
    {
      std::cout << solver_name() << "  Warning: non-zero velocity V_z=" << V_z
                << " m/s specified in transient simulations\n";
    }
#endif

  for (int i = 1;; ++i)
    {
      prepare_for_solve();
      assemble_system();
      solve_system();

      temperature.add(prm.get_double("Newton step length"), temperature_update);

      add_output("nNewton", i);

      // Check convergence
      const double max_abs_dT = temperature_update.linfty_norm();

      std::cout.unsetf(std::ios_base::floatfield);
      std::cout << std::setprecision(8);
      std::cout << solver_name() << "  "
                << "Time " << t << " s"
                << " step " << dt << " s"
                << "  Newton iteration " << i << "  max T change " << max_abs_dT
                << " K\n";
      if (max_abs_dT < prm.get_double("Max absolute change"))
        break;

      const int N = prm.get_integer("Max Newton iterations");
      if (N >= 1 && i >= N)
        break;
    }

  output_probes();

  if (dt > 0 && t + 1e-4 * dt >= t_max)
    return false;

  return dt > 0;
}

template <int dim>
const Triangulation<dim> &
TemperatureSolver<dim>::get_mesh() const
{
  return triangulation;
}

template <int dim>
Triangulation<dim> &
TemperatureSolver<dim>::get_mesh()
{
  return triangulation;
}

template <int dim>
const Vector<double> &
TemperatureSolver<dim>::get_temperature() const
{
  return temperature;
}

template <int dim>
Vector<double> &
TemperatureSolver<dim>::get_temperature()
{
  return temperature;
}

template <int dim>
const Vector<double> &
TemperatureSolver<dim>::get_heat_source() const
{
  return vol_heat_source;
}

template <int dim>
Vector<double> &
TemperatureSolver<dim>::get_heat_source()
{
  return vol_heat_source;
}

template <int dim>
double
TemperatureSolver<dim>::get_time() const
{
  return current_time;
}

template <int dim>
double &
TemperatureSolver<dim>::get_time()
{
  return current_time;
}

template <int dim>
double
TemperatureSolver<dim>::get_time_step() const
{
  return current_time_step;
}

template <int dim>
double &
TemperatureSolver<dim>::get_time_step()
{
  return current_time_step;
}

template <int dim>
double
TemperatureSolver<dim>::get_max_time() const
{
  return prm.get_double("Max time");
}

template <int dim>
const ParameterHandler &
TemperatureSolver<dim>::get_parameters() const
{
  return prm;
}

template <int dim>
ParameterHandler &
TemperatureSolver<dim>::get_parameters()
{
  return prm;
}

template <int dim>
void
TemperatureSolver<dim>::initialize()
{
  Timer timer;

  std::cout << solver_name() << "  Initializing finite element solution";

  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();

  temperature.reinit(n_dofs);
  temperature_update.reinit(n_dofs);
  vol_heat_source.reinit(n_dofs);

  std::cout << " " << format_time(timer) << "\n";

  std::cout << solver_name() << "  "
            << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
            << solver_name() << "  "
            << "Number of degrees of freedom for temperature: " << n_dofs
            << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::get_boundary_points(
  const unsigned int       id,
  std::vector<Point<dim>> &points,
  std::vector<bool> &      boundary_dofs) const
{
  get_support_points(points);
  boundary_dofs.resize(dh.n_dofs());
  DoFTools::extract_boundary_dofs(dh,
                                  ComponentMask(),
                                  boundary_dofs,
                                  {static_cast<types::boundary_id>(id)});
}

template <int dim>
void
TemperatureSolver<dim>::get_support_points(
  std::vector<Point<dim>> &points) const
{
  points.resize(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dh, points);
}

template <int dim>
std::map<unsigned int, Point<dim>>
TemperatureSolver<dim>::get_boundary_dofs(const unsigned int boundary_id) const
{
  std::vector<Point<dim>> all_points;
  std::vector<bool>       boundary_dofs;

  get_boundary_points(boundary_id, all_points, boundary_dofs);

  std::map<unsigned int, Point<dim>> boundary_points;
  for (unsigned int i = 0; i < all_points.size(); ++i)
    {
      if (boundary_dofs[i])
        boundary_points[i] = all_points[i];
    }
  return boundary_points;
}

template <int dim>
std::vector<double>
TemperatureSolver<dim>::get_field_at_points(
  const Vector<double> &         source,
  const std::vector<Point<dim>> &points) const
{
  Functions::FEFieldFunction<dim> ff(get_dof_handler(), source);

  std::vector<double> values(points.size(), 0);

  try
    {
      ff.value_list(points, values);
    }
  catch (std::exception &e)
    {
#ifdef DEBUG
      std::cout << e.what() << "\n";
#endif

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          try
            {
              values[i] = ff.value(points[i]);
            }
          catch (std::exception &e)
            {
              values[i] = 0;
            }
        }
    }

  return values;
}

template <int dim>
unsigned int
TemperatureSolver<dim>::get_degree() const
{
  return fe.degree;
}

template <int dim>
const DoFHandler<dim> &
TemperatureSolver<dim>::get_dof_handler() const
{
  return dh;
}

template <int dim>
void
TemperatureSolver<dim>::apply_bc1()
{
  for (const auto &bc : bc1_data)
    {
      const auto &T1 = bc.second;

      if (bc.second.size() == 1)
        {
          // homogeneous field
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dh,
                                                   bc.first,
                                                   ConstantFunction<dim>(T1[0]),
                                                   boundary_values);

          for (const auto &bv : boundary_values)
            temperature[bv.first] = bv.second;
        }
      else
        {
          const auto N = temperature.size();
          AssertThrow(N == T1.size(),
                      ExcMessage("TemperatureSolver: temperature size " +
                                 std::to_string(N) +
                                 " does not match BC1 size " +
                                 std::to_string(T1.size())));

          std::vector<bool> boundary_dofs(N, false);
          DoFTools::extract_boundary_dofs(dh,
                                          ComponentMask(),
                                          boundary_dofs,
                                          {static_cast<types::boundary_id>(
                                            bc.first)});

          for (unsigned int i = 0; i < N; ++i)
            {
              if (boundary_dofs[i])
                temperature[i] = T1[i];
            }
        }
    }
}

template <int dim>
void
TemperatureSolver<dim>::clear_bcs()
{
  bc1_data.clear();
  bc_rad_mixed_data.clear();
  bc_convective_data.clear();
}

template <int dim>
void
TemperatureSolver<dim>::set_bc1(const unsigned int id, const double val)
{
  bc1_data[id].reinit(1);
  bc1_data[id][0] = val;
}

template <int dim>
void
TemperatureSolver<dim>::set_bc1(const unsigned int    id,
                                const Vector<double> &val)
{
  bc1_data[id] = val;
}

template <int dim>
void
TemperatureSolver<dim>::set_bc_rad_mixed(
  const unsigned int                  id,
  const Vector<double> &              q_in,
  std::function<double(const double)> emissivity,
  std::function<double(const double)> emissivity_deriv,
  const double                        T_amb)
{
  bc_rad_mixed_data[id].q_in             = q_in;
  bc_rad_mixed_data[id].emissivity       = emissivity;
  bc_rad_mixed_data[id].emissivity_deriv = emissivity_deriv;
  bc_rad_mixed_data[id].T_amb            = T_amb;
}

template <int dim>
void
TemperatureSolver<dim>::set_bc_convective(const unsigned int id,
                                          const double       h,
                                          const double       T_ref)
{
  bc_convective_data[id].h     = h;
  bc_convective_data[id].T_ref = T_ref;
}

template <int dim>
void
TemperatureSolver<dim>::add_probe(const Point<dim> &p)
{
  probes.push_back(p);
}

template <int dim>
void
TemperatureSolver<dim>::add_output(const std::string &name, const double value)
{
  if (probes_header_written &&
      additional_output.find(name) == additional_output.end())
    throw std::runtime_error(solver_name() + "  add_output: cannot add '" +
                             name + "' which was not present at t=0");

  additional_output[name] = value;
}

template <int dim>
void
TemperatureSolver<dim>::add_field(const std::string &   name,
                                  const Vector<double> &value)
{
  additional_fields[name] = value;
}

template <int dim>
void
TemperatureSolver<dim>::load_data()
{
  Timer timer;

  read_data(get_temperature(), "temperature" + output_name_suffix());

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::output_data() const
{
  Timer timer;

  write_data(get_temperature(), "temperature" + output_name_suffix());

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::output_vtk() const
{
  Timer timer;

  const std::string file_name =
    "result-temperature" + output_name_suffix() + ".vtk";
  std::cout << solver_name() << "  Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(temperature, "T");
  data_out.add_data_vector(temperature_update, "dT");
  data_out.add_data_vector(vol_heat_source, "dot_q");

  Vector<double> rho(temperature.size()), c_p(temperature.size()),
    l(temperature.size()), dl_dT(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    {
      rho[i]   = calc_rho(temperature[i]);
      c_p[i]   = calc_c_p(temperature[i]);
      l[i]     = calc_lambda(temperature[i]);
      dl_dT[i] = calc_derivative_lambda(temperature[i]);
    }
  data_out.add_data_vector(rho, "rho");
  data_out.add_data_vector(c_p, "c_p");
  data_out.add_data_vector(l, "lambda");
  data_out.add_data_vector(dl_dT, "derivative_lambda");

  for (const auto &it : additional_fields)
    {
      if (it.second.size() == temperature.size())
        data_out.add_data_vector(it.second, it.first);
    }

  std::map<unsigned int, Vector<double>> q_rad, emissivity;
  for (const auto &data : bc_rad_mixed_data)
    {
      Vector<double> &q = q_rad[data.first];
      Vector<double> &e = emissivity[data.first];

      q.reinit(temperature);
      e.reinit(temperature);

      std::vector<bool> boundary_dofs(temperature.size(), false);
      DoFTools::extract_boundary_dofs(dh,
                                      ComponentMask(),
                                      boundary_dofs,
                                      {static_cast<types::boundary_id>(
                                        data.first)});

      for (unsigned int i = 0; i < temperature.size(); ++i)
        {
          if (!boundary_dofs[i])
            continue;
          e[i] = data.second.emissivity(temperature[i]);
          q[i] = sigma_SB * e[i] * std::pow(temperature[i], 4);
        }

      data_out.add_data_vector(data.second.q_in,
                               "q_in_" + std::to_string(data.first));
      data_out.add_data_vector(q, "q_rad_" + std::to_string(data.first));
      data_out.add_data_vector(e, "emissivity_" + std::to_string(data.first));
    }

  std::map<unsigned int, Vector<double>> q_conv;
  for (const auto &data : bc_convective_data)
    {
      Vector<double> &q = q_conv[data.first];

      q.reinit(temperature);

      std::vector<bool> boundary_dofs(temperature.size(), false);
      DoFTools::extract_boundary_dofs(dh,
                                      ComponentMask(),
                                      boundary_dofs,
                                      {static_cast<types::boundary_id>(
                                        data.first)});

      for (unsigned int i = 0; i < temperature.size(); ++i)
        {
          if (!boundary_dofs[i])
            continue;
          q[i] = data.second.h * (temperature[i] - data.second.T_ref);
        }

      data_out.add_data_vector(q, "q_conv_" + std::to_string(data.first));
    }

  data_out.build_patches(prm.get_integer("Output subdivisions"));

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  data_out.write_vtk(output);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::output_boundary_values(const unsigned int id) const
{
  Timer timer;

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  get_boundary_points(id, points, boundary_dofs);

  if (std::none_of(boundary_dofs.cbegin(),
                   boundary_dofs.cend(),
                   [](const bool b) { return b; }))
    {
      std::cout << solver_name()
                << "  output_boundary_values: skipping empty boundary " << id
                << "\n";
      return;
    }

  const std::string file_name = "result-temperature" + output_name_suffix() +
                                "-boundary" + std::to_string(id) + ".dat";
  std::cout << solver_name() << "  Saving to '" << file_name << "'";

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  const Vector<double> &T = get_temperature();

  const auto it_q_in = bc_rad_mixed_data.find(id);

  const auto dims = coordinate_names(dim);
  for (const auto &d : dims)
    output << d << "[m]\t";

  output << "T[K]\t"
         << "rho[kgm^-3]\t"
         << "c_p[Jkg^-1K^-1]\t"
         << "lambda[Wm^-1K^-1]";

  for (const auto &it : additional_fields)
    {
      if (it.second.size() == T.size())
        output << '\t' << it.first;
    }

  if (it_q_in != bc_rad_mixed_data.end())
    output << '\t' << "q_in[Wm^-2]";

  output << '\n';


  for (unsigned int i = 0; i < points.size(); ++i)
    {
      if (!boundary_dofs[i])
        continue;

      // a simple '<< points[i]' would put space between coordinates
      for (unsigned int d = 0; d < dim; ++d)
        output << points[i][d] << '\t';

      output << T[i] << '\t' << calc_rho(T[i]) << '\t' << calc_c_p(T[i]) << '\t'
             << calc_lambda(T[i]);

      for (const auto &it : additional_fields)
        {
          if (it.second.size() == T.size())
            output << '\t' << it.second[i];
        }

      if (it_q_in != bc_rad_mixed_data.end())
        output << '\t' << it_q_in->second.q_in[i];

      output << '\n';
    }

  std::cout << " " << format_time(timer) << "\n";

#ifdef DEBUG
  if (it_q_in != bc_rad_mixed_data.end())
    {
      const QGauss<dim - 1> face_quadrature(
        prm.get_integer("Number of face quadrature points"));
      FEFaceValues<dim> fe_face_values(
        fe, face_quadrature, update_quadrature_points | update_values);

      output_boundary_field_at_quadrature_points(get_dof_handler(),
                                                 fe_face_values,
                                                 it_q_in->second.q_in,
                                                 id,
                                                 "result-q_in" +
                                                   output_name_suffix() +
                                                   "-boundary" +
                                                   std::to_string(id) + ".dat");
    }
#endif
}

template <int dim>
void
TemperatureSolver<dim>::output_mesh() const
{
  Timer timer;

  const std::string file_name = "mesh" + output_name_suffix() + ".msh";
  std::cout << solver_name() << "  Saving to '" << file_name << "'";

  std::ofstream output(file_name);
  output << std::setprecision(16);

  GridOut grid_out;
  grid_out.set_flags(GridOutFlags::Msh(true));
  grid_out.write_msh(triangulation, output);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::output_parameter_table(const double       T1,
                                               const double       T2,
                                               const unsigned int n) const
{
  const std::string fname = "temperature-parameter-table.tsv";
  std::cout << solver_name() << "  Saving table '" << fname << "', T=" << T1
            << "-" << T2 << " K, n=" << n << '\n';

  std::ofstream output(fname);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  output << "T[K]\t"
         << "rho[kgm^-3]\t"
         << "c_p[Jkg^-1K^-1]\t"
         << "lambda[Wm^-1K^-1]\t"
         << "derivative_lambda[Wm^-1K^-2]\n";

  for (unsigned int i = 0; i < n; ++i)
    {
      const double T = T1 + (T2 - T1) * i / (n - 1);

      output << T << '\t' << calc_rho(T) << '\t' << calc_c_p(T) << '\t'
             << calc_lambda(T) << '\t' << calc_derivative_lambda(T) << '\n';
    }
}

template <int dim>
double
TemperatureSolver<dim>::calc_lambda(const double T) const
{
  return m_lambda.value(Point<1>(T));
}

template <int dim>
double
TemperatureSolver<dim>::calc_derivative_lambda(const double T) const
{
  return m_derivative_lambda.value(Point<1>(T));
}

template <int dim>
double
TemperatureSolver<dim>::calc_rho(const double T) const
{
  return m_rho.value(Point<1>(T));
}

template <int dim>
double
TemperatureSolver<dim>::calc_c_p(const double T) const
{
  return m_c_p.value(Point<1>(T));
}

template <int dim>
double
TemperatureSolver<dim>::calc_rho_c_p(const double T) const
{
  const Point<1> x(T);
  return m_rho.value(x) * m_c_p.value(x);
}

template <int dim>
double
TemperatureSolver<dim>::calc_V_z() const
{
  return prm.get_double("Velocity");
}

template <int dim>
std::string
TemperatureSolver<dim>::output_name_suffix() const
{
  std::stringstream ss;
  ss << std::setprecision(8);
  ss << "-" << dim << "d-order" << get_degree() << "-t" << get_time();
  return ss.str();
}

template <int dim>
void
TemperatureSolver<dim>::output_probes() const
{
  Timer timer;

  std::stringstream ss;
  ss << "probes-temperature-" << dim << "d.txt";
  const std::string file_name = ss.str();

  std::cout << solver_name() << "  "
            << "Saving values at probe points to '" << file_name << "'";

  const unsigned int N = probes.size();

  const double t  = get_time();
  const double dt = get_time_step();

  if (!probes_header_written)
    {
      // write header at the first time step
      std::ofstream output(file_name);

      for (unsigned int i = 0; i < N; ++i)
        output << "# probe " << i << ":\t" << probes[i] << "\n";

      output << "t[s]"
             << "\tdt[s]";

      for (const auto &it : additional_output)
        output << "\t" << it.first;

      output << "\tT_min[K]"
             << "\tT_max[K]";
      for (unsigned int i = 0; i < N; ++i)
        output << "\tT_" << i << "[K]";
      for (unsigned int i = 0; i < N; ++i)
        output << "\trho_" << i << "[kgm^-3]";
      for (unsigned int i = 0; i < N; ++i)
        output << "\tc_p_" << i << "[Jkg^-1K^-1]";
      for (unsigned int i = 0; i < N; ++i)
        output << "\tlambda_" << i << "[Wm^-1K^-1]";
      for (unsigned int i = 0; i < N; ++i)
        output << "\tderivative_lambda_" << i << "[Wm^-1K^-2]";
      output << "\n";
    }

  const std::vector<double> values = get_field_at_probes(temperature);

  const auto limits = minmax(temperature);

  // header is already written, append values at the current time step
  std::ofstream output(file_name, std::ios::app);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  output << t << '\t' << dt;

  for (const auto &it : additional_output)
    output << "\t" << it.second;

  output << '\t' << limits.first << '\t' << limits.second;

  for (const auto &v : values)
    output << '\t' << v;
  for (const auto &v : values)
    output << '\t' << calc_rho(v);
  for (const auto &v : values)
    output << '\t' << calc_c_p(v);
  for (const auto &v : values)
    output << '\t' << calc_lambda(v);
  for (const auto &v : values)
    output << '\t' << calc_derivative_lambda(v);
  output << "\n";

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
std::vector<double>
TemperatureSolver<dim>::get_field_at_probes(const Vector<double> &source) const
{
  return get_field_at_points(source, probes);
}

template <int dim>
void
TemperatureSolver<dim>::prepare_for_solve()
{
  const unsigned int n_dofs = dh.n_dofs();

  temperature_update.reinit(n_dofs);
  system_rhs.reinit(n_dofs);

  // Apply Dirichlet boundary conditions
  apply_bc1();

  if (!sparsity_pattern.empty() && !system_matrix.empty())
    return;

  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_sparsity_pattern(dh, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
TemperatureSolver<dim>::AssemblyScratchData::AssemblyScratchData(
  const Quadrature<dim>     quadrature,
  const Quadrature<dim - 1> face_quadrature,
  const FiniteElement<dim> &fe)
  : fe_values(fe,
              quadrature,
              update_quadrature_points | update_values | update_gradients |
                update_JxW_values)
  , fe_face_values(fe,
                   face_quadrature,
                   update_quadrature_points | update_values | update_JxW_values)
  , T_q(quadrature.size())
  , T_prev_q(quadrature.size())
  , dot_q_q(quadrature.size())
  , grad_T_q(quadrature.size())
  , T_face_q(face_quadrature.size())
  , q_in_face_q(face_quadrature.size())
{}

template <int dim>
TemperatureSolver<dim>::AssemblyScratchData::AssemblyScratchData(
  const AssemblyScratchData &scratch_data)
  : fe_values(scratch_data.fe_values.get_fe(),
              scratch_data.fe_values.get_quadrature(),
              scratch_data.fe_values.get_update_flags())
  , fe_face_values(scratch_data.fe_face_values.get_fe(),
                   scratch_data.fe_face_values.get_quadrature(),
                   scratch_data.fe_face_values.get_update_flags())
  , T_q(scratch_data.T_q)
  , T_prev_q(scratch_data.T_prev_q)
  , dot_q_q(scratch_data.dot_q_q)
  , grad_T_q(scratch_data.grad_T_q)
  , T_face_q(scratch_data.T_face_q)
  , q_in_face_q(scratch_data.q_in_face_q)
{}

template <int dim>
void
TemperatureSolver<dim>::assemble_system()
{
  Timer timer;

  std::cout << solver_name() << "  Assembling system";

  const QGauss<dim> quadrature(
    prm.get_integer("Number of cell quadrature points"));
  const QGauss<dim - 1> face_quadrature(
    prm.get_integer("Number of face quadrature points"));

  system_matrix = 0;
  system_rhs    = 0;

  WorkStream::run(dh.begin_active(),
                  dh.end(),
                  *this,
                  &TemperatureSolver::local_assemble_system,
                  &TemperatureSolver::copy_local_to_global,
                  AssemblyScratchData(quadrature, face_quadrature, fe),
                  AssemblyCopyData());

  // Apply boundary conditions for Newton update
  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &bc : bc1_data)
    {
      VectorTools::interpolate_boundary_values(dh,
                                               bc.first,
                                               ZeroFunction<dim>(),
                                               boundary_values);
    }

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     temperature_update,
                                     system_rhs);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::local_assemble_system(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  AssemblyScratchData &                                 scratch_data,
  AssemblyCopyData &                                    copy_data)
{
  // precalculate constant parameters
  const double dt     = get_time_step();
  const double inv_dt = dt == 0 ? 0 : 1 / dt;
  const double V_z    = calc_V_z();

  FEValues<dim> &            fe_values       = scratch_data.fe_values;
  FEFaceValues<dim> &        fe_face_values  = scratch_data.fe_face_values;
  const Quadrature<dim> &    quadrature      = fe_values.get_quadrature();
  const Quadrature<dim - 1> &face_quadrature = fe_face_values.get_quadrature();

  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int n_q_points      = quadrature.size();
  const unsigned int n_face_q_points = face_quadrature.size();


  FullMatrix<double> &cell_matrix = copy_data.cell_matrix;
  Vector<double> &    cell_rhs    = copy_data.cell_rhs;

  // resize and initialize with zeros
  cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  cell_rhs.reinit(dofs_per_cell);

  std::vector<double> &        T_q      = scratch_data.T_q;
  std::vector<double> &        T_prev_q = scratch_data.T_prev_q;
  std::vector<double> &        dot_q_q  = scratch_data.dot_q_q;
  std::vector<Tensor<1, dim>> &grad_T_q = scratch_data.grad_T_q;

  std::vector<types::global_dof_index> &local_dof_indices =
    copy_data.local_dof_indices;

  local_dof_indices.resize(dofs_per_cell);


  fe_values.reinit(cell);

  fe_values.get_function_values(temperature, T_q);
  fe_values.get_function_values(temperature_prev, T_prev_q);
  fe_values.get_function_values(vol_heat_source, dot_q_q);
  fe_values.get_function_gradients(temperature, grad_T_q);

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double rho_c_p = calc_rho_c_p(T_q[q]);
      const double tmp     = inv_dt * rho_c_p;
      const double conv_q  = rho_c_p * V_z * grad_T_q[q][dim - 1];

      const double lambda_q       = calc_lambda(T_q[q]);
      const double lambda_deriv_q = calc_derivative_lambda(T_q[q]);

      const double weight =
        dim == 2 ? fe_values.JxW(q) * fe_values.quadrature_point(q)[0] :
                   fe_values.JxW(q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double &        phi_i      = fe_values.shape_value(i, q);
          const Tensor<1, dim> &grad_phi_i = fe_values.shape_grad(i, q);

          // hand-written optimizations
          const double tmp_i      = tmp * phi_i;
          const double tmp_grad_i = grad_T_q[q] * grad_phi_i;
          const double tmp2_i     = lambda_deriv_q * tmp_grad_i + tmp_i;
          const double tmp_V_i    = rho_c_p * V_z * phi_i;

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const double &        phi_j      = fe_values.shape_value(j, q);
              const Tensor<1, dim> &grad_phi_j = fe_values.shape_grad(j, q);

              // Newthon's method
              cell_matrix(i, j) +=
                (lambda_q * (grad_phi_j * grad_phi_i) + tmp2_i * phi_j +
                 tmp_V_i * grad_phi_j[dim - 1]) *
                weight;
            }

          cell_rhs(i) -=
            (lambda_q * tmp_grad_i + tmp_i * (T_q[q] - T_prev_q[q]) +
             (conv_q - dot_q_q[q]) * phi_i) *
            weight;
        }
    }

  for (unsigned int face_number = 0;
       face_number < GeometryInfo<dim>::faces_per_cell;
       ++face_number)
    {
      if (!cell->face(face_number)->at_boundary())
        continue;

      const auto it =
        bc_rad_mixed_data.find(cell->face(face_number)->boundary_id());
      if (it == bc_rad_mixed_data.end())
        continue;

      std::vector<double> &T_face_q    = scratch_data.T_face_q;
      std::vector<double> &q_in_face_q = scratch_data.q_in_face_q;

      fe_face_values.reinit(cell, face_number);

      fe_face_values.get_function_values(temperature, T_face_q);
      fe_face_values.get_function_values(it->second.q_in, q_in_face_q);

      for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          const double net_heat_flux =
            sigma_SB * it->second.emissivity(T_face_q[q]) *
              (std::pow(T_face_q[q], 4) - std::pow(it->second.T_amb, 4)) -
            q_in_face_q[q];

          const double weight =
            dim == 2 ?
              fe_face_values.JxW(q) * fe_face_values.quadrature_point(q)[0] :
              fe_face_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) +=
                    sigma_SB *
                    (it->second.emissivity(T_face_q[q]) * 4.0 *
                       std::pow(T_face_q[q], 3) +
                     it->second.emissivity_deriv(T_face_q[q]) *
                       (std::pow(T_face_q[q], 4) -
                        std::pow(it->second.T_amb, 4))) *
                    fe_face_values.shape_value(i, q) *
                    fe_face_values.shape_value(j, q) * weight;
                }
              cell_rhs(i) -=
                net_heat_flux * fe_face_values.shape_value(i, q) * weight;
            }
        }
    }

  for (unsigned int face_number = 0;
       face_number < GeometryInfo<dim>::faces_per_cell;
       ++face_number)
    {
      if (!cell->face(face_number)->at_boundary())
        continue;

      const auto it =
        bc_convective_data.find(cell->face(face_number)->boundary_id());
      if (it == bc_convective_data.end())
        continue;

      const double h     = it->second.h;
      const double T_ref = it->second.T_ref;

      std::vector<double> &T_face_q = scratch_data.T_face_q;

      fe_face_values.reinit(cell, face_number);

      fe_face_values.get_function_values(temperature, T_face_q);

      for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          const double net_heat_flux = h * (T_face_q[q] - T_ref);

          const double weight =
            dim == 2 ?
              fe_face_values.JxW(q) * fe_face_values.quadrature_point(q)[0] :
              fe_face_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += h * fe_face_values.shape_value(i, q) *
                                       fe_face_values.shape_value(j, q) *
                                       weight;
                }
              cell_rhs(i) -=
                net_heat_flux * fe_face_values.shape_value(i, q) * weight;
            }
        }
    }

  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void
TemperatureSolver<dim>::copy_local_to_global(const AssemblyCopyData &copy_data)
{
  for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
    {
      for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
        system_matrix.add(copy_data.local_dof_indices[i],
                          copy_data.local_dof_indices[j],
                          copy_data.cell_matrix(i, j));
      system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
    }
}

template <int dim>
void
TemperatureSolver<dim>::advance_time()
{
  double &     dt     = get_time_step();
  const double t_prev = get_time();
  const double t_max  = get_max_time();

  // stop exactly at max time
  if (t_prev + (1 + 1e-4) * dt >= t_max)
    dt = t_max - t_prev;

  get_time() += get_time_step();

  temperature_prev = temperature;
}

template <int dim>
void
TemperatureSolver<dim>::solve_system()
{
  Timer timer;

  std::cout << solver_name() << "  Solving system";

  const std::string solver_type = prm.get("Linear solver type");

  if (solver_type == "UMFPACK")
    {
      std::cout << " (" << solver_type << ")";

      SparseDirectUMFPACK A;
      A.initialize(system_matrix);
      A.vmult(temperature_update, system_rhs);
    }
  else
    {
      const unsigned int solver_iterations =
        prm.get_integer("Linear solver iterations");
      const double solver_tolerance = prm.get_double("Linear solver tolerance");

      const bool log_history = prm.get_bool("Log convergence full");
      const bool log_result  = prm.get_bool("Log convergence final");

      if (log_history || log_result)
        std::cout << "\n";

      IterationNumberControl control(solver_iterations,
                                     solver_tolerance,
                                     log_history,
                                     log_result);

      SolverSelector<> solver;
      solver.select(solver_type);
      solver.set_control(control);

      const std::string preconditioner_type = prm.get("Preconditioner type");
      const double      preconditioner_relaxation =
        prm.get_double("Preconditioner relaxation");

      PreconditionSelector<> preconditioner(preconditioner_type,
                                            preconditioner_relaxation);
      preconditioner.use_matrix(system_matrix);

      solver.solve(system_matrix,
                   temperature_update,
                   system_rhs,
                   preconditioner);

      if (control.last_step() >= solver_iterations ||
          control.last_value() >= solver_tolerance)
        {
          if (!(log_history || log_result))
            std::cout << "\n";

          std::cout << solver_name() << "  Warning: not converged! Residual(0)="
                    << control.initial_value() << " Residual("
                    << control.last_step() << ")=" << control.last_value()
                    << "\n";
        }
    }

  std::cout << " " << format_time(timer) << "\n";
}

#endif