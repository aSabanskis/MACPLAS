#ifndef macplas_stress_solver_h
#define macplas_stress_solver_h

#include <deal.II/base/function_parser.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "utilities.h"

using namespace dealii;

#if DEAL_II_VERSION_GTE(9, 5, 0)
template <int dim, typename RangeNumberType = double>
using ConstantFunction = Functions::ConstantFunction<dim, RangeNumberType>;

template <int dim, typename RangeNumberType = double>
using ZeroFunction = Functions::ZeroFunction<dim, RangeNumberType>;
#endif

/** Class for calculation of the thermal stresses for a given temperature field.
 * Implemented for \c dim=2 (axisymmetric) and 3 (three-dimensional simulation).
 * The creep strain is also taken into account (updated by DislocationSolver).
 */
template <int dim>
class StressSolver
{
public:
  /** Constructor.
   * Initializes the solver parameters from \c stress.prm.
   * If it doesn't exist, the default parameter values are written to
   * \c stress-default.prm.
   * Default values are used and written to \c stress.prm if
   * \c use_default_prm parameter is specified.
   */
  explicit StressSolver(const unsigned int order           = 2,
                        const bool         use_default_prm = false);

  /** Solver name
   */
  std::string
  solver_name() const;

  /** Calculate the stress field.
   * Only postprocesses the existing displacement field is \c postprocess_only
   * is enabled (faster).
   */
  void
  solve(const bool postprocess_only = false);

  /** Check if the simulation has converged
   */
  bool
  has_converged() const;

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

  /** Get displacement \f$\mathbf{u}\f$, m
   */
  const BlockVector<double> &
  get_displacement() const;

  /** Get displacement \f$\mathbf{u}\f$, m
   */
  BlockVector<double> &
  get_displacement();

  /** Get stress \f$\sigma_{ij}\f$, Pa
   */
  const BlockVector<double> &
  get_stress() const;

  /** Get stress deviator \f$S_{ij} =
   * \sigma_{ij} - \frac{1}{3} \delta_{ij} \sigma_{kk}\f$, Pa
   */
  const BlockVector<double> &
  get_stress_deviator() const;

  /** Get mean (hydrostatic) stress \f$\sigma_\mathrm{ave} =
   * \frac{1}{3} \sigma_{kk}\f$, Pa
   */
  const Vector<double> &
  get_stress_hydrostatic() const;

  /** Get second invariant of deviatoric stress \f$J_2 =
   * \frac{1}{2} S_{ij} S_{ij}\f$, Pa<sup>2</sup>
   */
  const Vector<double> &
  get_stress_J_2() const;

  /** Get elastic strain \f$\varepsilon^e_{ij}\f$, dimensionless
   */
  const BlockVector<double> &
  get_strain_e() const;

  /** Get creep strain \f$\varepsilon^c_{ij}\f$, dimensionless
   */
  const BlockVector<double> &
  get_strain_c() const;

  /** Get creep strain \f$\varepsilon^c_{ij}\f$, dimensionless
   */
  BlockVector<double> &
  get_strain_c();

  /** Calculate the temperature-dependent Young's modulus \f$E\f$, Pa
   */
  double
  calc_E(const double T) const;

  /** Calculate the temperature-dependent thermal expansion coefficient
   * \f$\alpha\f$, K<sup>-1</sup>
   */
  double
  calc_alpha(const double T) const;

  /** Get parameters StressSolver::prm
   */
  const ParameterHandler &
  get_parameters() const;

  /** Get parameters StressSolver::prm
   */
  ParameterHandler &
  get_parameters();

  /** Initialize DOF handler and fields
   */
  void
  initialize();

  /** Get coordinates of DOFs
   */
  void
  get_support_points(std::vector<Point<dim>> &points) const;

  /** Get finite element degree
   */
  unsigned int
  get_degree() const;

  /** Get degrees of freedom for temperature
   */
  const DoFHandler<dim> &
  get_dof_handler() const;

  /** Set first-type boundary condition
   */
  void
  set_bc1(const unsigned int id,
          const unsigned int component,
          const double       val);

  /** Apply boundary load
   */
  void
  set_bc_load(const unsigned int id, const Tensor<1, dim> val);

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

  /** Calculate and write temperature-dependent parameters to disk
   */
  void
  output_parameter_table(const double       T1 = 250,
                         const double       T2 = 1700,
                         const unsigned int n  = 30) const;

  /** Number of distinct elements of the stress tensor (3D: 6, 2D: 4)
   */
  static const unsigned int n_components = 2 * dim;

  /** Names of stress components in Voigt notation
   */
  const std::vector<std::string>
  stress_component_names() const;

private:
  /** Method of calculation of the elastic matrix
   */
  enum ElasticMatrixType
  {
    Enu, ///< Isotropic (from Young's modulus and Poisson's ratio)
    Cij, ///< C_11, C_12 and C_44
    Full ///< Full matrix
  };

  /** Initialize elastic parameters and \c Cij_type. Called by
   * StressSolver::initialize_parameters
   */
  void
  initialize_elastic_parameters();

  /** Initialize all parameters. Called by the constructor
   */
  void
  initialize_parameters();

  /** Initialize data before calculation.
   * Called by StressSolver::solve.
   */
  void
  prepare_for_solve();

  /** Assemble the system matrix and right-hand-side vector (multithreaded).
   * Called by StressSolver::solve.
   */
  void
  assemble_system();

  /** Structure that holds scratch data
   */
  struct AssemblyScratchData
  {
    AssemblyScratchData(const Quadrature<dim> &   quadrature,
                        const Quadrature<dim - 1> face_quadrature,
                        const FiniteElement<dim> &fe_temp,
                        const FiniteElement<dim> &fe);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);

    FEValues<dim>     fe_values_temp;
    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;

    std::vector<double>              T_q;
    std::vector<std::vector<double>> epsilon_c_q;
  };

  /** Structure that holds local contributions
   */
  struct AssemblyCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };

  /** Iterator tuple
   */
  using IteratorTuple =
    std::tuple<typename DoFHandler<dim>::active_cell_iterator,
               typename DoFHandler<dim>::active_cell_iterator>;

  /** Iterator pair
   */
  typedef SynchronousIterators<IteratorTuple> IteratorPair;

  /** Local assembly function
   */
  void
  local_assemble_system(const IteratorPair & cell_pair,
                        AssemblyScratchData &scratch_data,
                        AssemblyCopyData &   copy_data);

  /** Copy local contributions to global
   */
  void
  copy_local_to_global(const AssemblyCopyData &copy_data);

  /** Solve the system of linear equations.
   * Called by StressSolver::solve.
   */
  void
  solve_system();

  /** Calculate the second-order elastic constant (stiffness) \f$C_{11} =
   * E (1 - \nu) / ((1 + \nu) (1 - 2 \nu))\f$, Pa
   */
  double
  calc_C_11(const double T) const;

  /** Calculate the second-order elastic constant (stiffness) \f$C_{12} =
   * E \nu / ((1 + \nu) (1 - 2 \nu))\f$, Pa
   */
  double
  calc_C_12(const double T) const;

  /** Calculate the second-order elastic constant (stiffness) \f$C_{44} =
   * E / (2 (1 + \nu))\f$, Pa
   */
  double
  calc_C_44(const double T) const;

  /** Calculate the anisotropic factor \f$H = 2 C_{44} + C_{12} − C_{11}\f$, Pa
   */
  double
  calc_H(const double T) const;

  /** Recover strain at DOFs via extrapolation from quadrature points.
   * Called by StressSolver::calculate_stress.
   */
  void
  recover_strain_extrapolation();

  /** Recover strain at DOFs via global projection.
   * Called by StressSolver::calculate_stress.
   */
  void
  recover_strain_global();

  /** Calculate stress from the displacement field.
   * Called by StressSolver::solve.
   */
  void
  calculate_stress(const bool skip_recovery = false);

  /** Helper function to calculate stress from strain.
   * Called by StressSolver::calculate_stress.
   */
  void
  calculate_stress_from_strain();

  /** Helper function to calculate stress invariants.
   * Called by StressSolver::calculate_stress.
   */
  void
  calculate_stress_invariants();

  /** Get stiffness tensor
   */
  SymmetricTensor<2, StressSolver<dim>::n_components>
  get_stiffness_tensor(const double &T) const;

  /** Get strain from \c FEValues
   */
  void
  get_strain(const FEValues<dim> &    fe_values,
             const unsigned int &     shape_func,
             const unsigned int &     q,
             Tensor<1, n_components> &strain) const;

  /** Get strain from temperature
   */
  void
  get_strain(const double &T, Tensor<1, n_components> &strain) const;

  /** Get strain from displacement
   */
  void
  get_strain(const Point<dim> &                 point_q,
             const Vector<double> &             displacement_q,
             const std::vector<Tensor<1, dim>> &grad_displacement,
             Tensor<1, n_components> &          strain) const;

  /** Helper method for creating output file name.
   * @returns \c "-<dim>d-order<order>"
   */
  std::string
  output_name_suffix() const;

  /** Mesh
   */
  Triangulation<dim> triangulation;

  /** Finite element for temperature
   */
  FE_Q<dim> fe_temp;

  /** Degrees of freedom for temperature
   */
  DoFHandler<dim> dh_temp;

  /** Temperature \f$T\f$, K
   */
  Vector<double> temperature;

  /** Finite element for displacement
   */
  FESystem<dim> fe;

  /** Degrees of freedom for displacement
   */
  DoFHandler<dim> dh;

  /** Displacement \f$\mathbf{u}\f$, m
   */
  BlockVector<double> displacement;

  /** Stress \f$\sigma_{ij}\f$, Pa
   */
  BlockVector<double> stress;

  /** Stress deviator \f$S_{ij}\f$, Pa
   */
  BlockVector<double> stress_deviator;

  /** Elastic strain \f$\varepsilon^e_{ij}\f$, dimensionless
   */
  BlockVector<double> strain_e;

  /** Creep strain \f$\varepsilon^c_{ij}\f$, dimensionless
   */
  BlockVector<double> strain_c;

  /** Mean (hydrostatic) stress \f$\sigma_\mathrm{ave} =
   * \frac{1}{3} \sigma_{kk}\f$, Pa
   */
  Vector<double> stress_hydrostatic;

  /** von Mises stress \f$\sigma_\mathrm{vM} =
   * \sqrt{3 J_2}\f$, Pa
   */
  Vector<double> stress_von_Mises;

  /** Second invariant of deviatoric stress \f$J_2\f$, Pa
   */
  Vector<double> stress_J_2;

  /** Sparsity pattern
   */
  BlockSparsityPattern sparsity_pattern;

  /** System matrix
   */
  BlockSparseMatrix<double> system_matrix;

  /** Right-hand-side vector
   */
  BlockVector<double> system_rhs;

  /** Flag for checking simulation success
   */
  bool converged;

  /** Data for first-type BC.
   * Map key: boundary id and component, value contains displacement.
   * This allows to apply BC to multiple components at the same boundary.
   */
  std::map<std::pair<unsigned int, unsigned int>, double> bc1_data;

  /** Data for load BC.
   * Map key: boundary id, value contains applied load.
   */
  std::map<unsigned int, Tensor<1, dim>> bc_load_data;

  /** Parameter handler
   */
  ParameterHandler prm;

  /** Method of calculation of the elastic constants
   */
  ElasticMatrixType Cij_type;

  /** Elastic constant (temperature function) \f$C_{11}\f$, Pa
   */
  FunctionParser<1> m_C_11;

  /** Elastic constant (temperature function) \f$C_{12}\f$, Pa
   */
  FunctionParser<1> m_C_12;

  /** Elastic constant (temperature function) \f$C_{44}\f$, Pa
   */
  FunctionParser<1> m_C_44;

  /** Full elastic constant matrix (temperature functions) \f$C_{ij}\f$, Pa
   */
  std::array<FunctionParser<1>, n_components * n_components> m_C_full;

  /** Young's modulus (temperature function) \f$E\f$, Pa
   */
  FunctionParser<1> m_E;

  /** Thermal expansion coefficient (temperature function) \f$\alpha\f$,
   * K<sup>-1</sup>
   */
  FunctionParser<1> m_alpha;

  /** Poisson's ratio \f$\nu\f$, dimensionless
   */
  double m_nu;

  /** Reference temperature \f$T_\mathrm{ref}\f$, K
   */
  double m_T_ref;
};


// IMPLEMENTATION

template <int dim>
StressSolver<dim>::StressSolver(const unsigned int order,
                                const bool         use_default_prm)
  : fe_temp(order)
  , dh_temp(triangulation)
  , fe(FE_Q<dim>(order), dim)
  , dh(triangulation)
  , converged(false)
  , Cij_type(ElasticMatrixType::Enu)
{
  std::cout << "Creating stress solver, order=" << order << ", dim=" << dim
            << " ("
#ifdef DEBUG
               "Debug"
#else
               "Release"
#endif
               ")\n";

  AssertThrow(dim == 2 || dim == 3, ExcNotImplemented());

  std::cout << "Stress components in Voigt notation:\n";
  const auto names = stress_component_names();
  for (unsigned int i = 0; i < names.size(); ++i)
    std::cout << i << " " << names[i] << "\n";

  const std::string info_T = " (temperature function)";

  prm.declare_entry("Elastic constant C_11",
                    "",
                    Patterns::Anything(),
                    "Optional elastic constant C_11 in Pa" + info_T);

  prm.declare_entry("Elastic constant C_12",
                    "",
                    Patterns::Anything(),
                    "Optional elastic constant C_12 in Pa" + info_T);

  prm.declare_entry("Elastic constant C_44",
                    "",
                    Patterns::Anything(),
                    "Optional elastic constant C_44 in Pa" + info_T);

  prm.declare_entry(
    "Elastic constant matrix",
    "",
    Patterns::Anything(),
    "Optional comma-separated full elastic constant matrix C_ij in Pa" +
      info_T);

  // Physical parameters from https://doi.org/10.1016/S0022-0248(01)01322-7
  prm.declare_entry("Young's modulus",
                    "1.56e11",
                    Patterns::Anything(),
                    "Young's modulus E in Pa" + info_T);

  prm.declare_entry("Thermal expansion coefficient",
                    "3.2e-6",
                    Patterns::Anything(),
                    "Thermal expansion coefficient alpha in 1/K" + info_T);

  prm.declare_entry("Poisson's ratio",
                    "0.25",
                    Patterns::Double(0, 0.5),
                    "Poisson's ratio nu (dimensionless)");

  prm.declare_entry("Reference temperature",
                    "1685",
                    Patterns::Double(0),
                    "Reference temperature T_ref in K");

  prm.enter_subsection("Stress recovery");
  {
    prm.declare_entry("Method",
                      "extrapolation",
                      Patterns::Selection("extrapolation|global"),
                      "Method for stress postprocessing"
                      " (extrapolation requires no additional parameters)");

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
  }
  prm.leave_subsection();

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
                    Patterns::Selection("none|jacobi"), // limited options
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

  prm.declare_entry("Output precision",
                    "8",
                    Patterns::Integer(1),
                    "Precision of double variables for output of field data");

  prm.declare_entry("Output subdivisions",
                    "0",
                    Patterns::Integer(0),
                    "Number of cell subdivisions for vtk output (0: order)");

  if (use_default_prm)
    {
      std::ofstream of("stress.prm");
      prm.print_parameters(of, ParameterHandler::Text);
    }
  else
    try
      {
        prm.parse_input("stress.prm");
      }
    catch (std::exception &e)
      {
        std::cout << e.what() << "\n";

        std::ofstream of("stress-default.prm");
        prm.print_parameters(of, ParameterHandler::Text);
      }

  initialize_parameters();
}

template <int dim>
void
StressSolver<dim>::initialize_elastic_parameters()
{
  const std::string m_E_expression = prm.get("Young's modulus");
  m_E.initialize("T", m_E_expression, typename FunctionParser<1>::ConstMap());
  m_nu = prm.get_double("Poisson's ratio");

  const std::string m_C_11_expression = prm.get("Elastic constant C_11");
  const std::string m_C_12_expression = prm.get("Elastic constant C_12");
  const std::string m_C_44_expression = prm.get("Elastic constant C_44");

  const bool use_elastic_constants =
    !(m_C_11_expression.empty() || m_C_12_expression.empty() ||
      m_C_44_expression.empty());

  if (use_elastic_constants)
    {
      Cij_type = ElasticMatrixType::Cij;

      m_C_11.initialize("T",
                        m_C_11_expression,
                        typename FunctionParser<1>::ConstMap());
      m_C_12.initialize("T",
                        m_C_12_expression,
                        typename FunctionParser<1>::ConstMap());
      m_C_44.initialize("T",
                        m_C_44_expression,
                        typename FunctionParser<1>::ConstMap());

      std::cout << "C_11=" << m_C_11_expression << "\n"
                << "C_12=" << m_C_12_expression << "\n"
                << "C_44=" << m_C_44_expression << "\n";

      return;
    }

  const std::string        Cij_s     = prm.get("Elastic constant matrix");
  std::vector<std::string> Cij_split = Utilities::split_string_list(Cij_s, ',');

#ifdef DEBUG
  std::cout << Cij_split.size() << ' ' << m_C_full.size() << '\n';
#endif

  const bool use_full_matix = Cij_split.size() == m_C_full.size();

  if (use_full_matix)
    {
      Cij_type = ElasticMatrixType::Full;

      std::array<int, n_components> colum_widths;

      for (unsigned int i = 0; i < n_components; ++i)
        {
          colum_widths[i] = 0;
          for (unsigned int j = 0; j < n_components; ++j)
            {
              std::string &Cij = Cij_split[i + j * n_components];
              colum_widths[i]  = std::max(colum_widths[i], (int)Cij.size());
            }
#ifdef DEBUG
          std::cout << i << ' ' << colum_widths[i] << '\n';
#endif
        }

      std::cout << "C_ij=\n";
      for (unsigned int j = 0; j < n_components; ++j)
        {
          for (unsigned int i = 0; i < n_components; ++i)
            {
              std::string &Cij = Cij_split[i + j * n_components];

              if (Cij.empty())
                Cij = "0";

              std::cout << std::setw(colum_widths[i]) << Cij;

              if (i + 1 < n_components)
                std::cout << ", ";
              else
                std::cout << '\n';
            }
        }

      for (unsigned int j = 0; j < n_components; ++j)
        {
          for (unsigned int i = 0; i <= j; ++i)
            {
              const std::string &Cij = Cij_split[i + j * n_components];
              const std::string &Cji = Cij_split[j + i * n_components];

              AssertThrow(Cij == Cji,
                          ExcMessage("C_" + std::to_string(i) +
                                     std::to_string(j) + " = '" + Cij +
                                     "' not equal to C_" + std::to_string(j) +
                                     std::to_string(i) + " = '" + Cji + "'"));

              // lower triangle will be uninitialized
              m_C_full[i + j * n_components].initialize(
                "T", Cij, typename FunctionParser<1>::ConstMap());
            }
        }

      return;
    }

  std::cout << "E=" << m_E_expression << "\n"
            << "nu=" << m_nu << "\n";
}

template <int dim>
void
StressSolver<dim>::initialize_parameters()
{
  std::cout << solver_name() << "  Initializing parameters\n";

  initialize_elastic_parameters();

  const std::string m_alpha_expression =
    prm.get("Thermal expansion coefficient");
  m_alpha.initialize("T",
                     m_alpha_expression,
                     typename FunctionParser<1>::ConstMap());

  m_T_ref = prm.get_double("Reference temperature");

  const auto n_threads = prm.get_integer("Number of threads");
  MultithreadInfo::set_thread_limit(n_threads > 0 ? n_threads :
                                                    MultithreadInfo::n_cores());

  const long int n_q_default = get_degree() + 1;

  if (prm.get_integer("Number of cell quadrature points") == 0)
    prm.set("Number of cell quadrature points", n_q_default);

  if (prm.get_integer("Number of face quadrature points") == 0)
    prm.set("Number of face quadrature points", n_q_default);

  prm.enter_subsection("Stress recovery");
  if (prm.get_integer("Number of cell quadrature points") == 0)
    prm.set("Number of cell quadrature points", n_q_default);
  prm.leave_subsection();

  const long int n_vtk_default = get_degree();

  if (prm.get_integer("Output subdivisions") == 0)
    prm.set("Output subdivisions", n_vtk_default);

  std::cout << "alpha=" << m_alpha_expression << "\n"
            << "T_ref=" << m_T_ref << "\n";

  std::cout << "n_q_cell=" << prm.get("Number of cell quadrature points")
            << "\n"
            << "n_q_face=" << prm.get("Number of face quadrature points")
            << "\n";

  std::cout << "n_cores=" << MultithreadInfo::n_cores() << "\n"
            << "n_threads=" << MultithreadInfo::n_threads() << "\n";
}

template <int dim>
std::string
StressSolver<dim>::solver_name() const
{
  return "MACPLAS:Stress";
}

template <int dim>
void
StressSolver<dim>::solve(const bool postprocess_only)
{
  converged = true;

  if (postprocess_only)
    {
      calculate_stress(true);
      return;
    }

  prepare_for_solve();
  assemble_system();
  try
    {
      solve_system();
      calculate_stress();
    }
  catch (std::exception &e)
    {
      std::cout << '\n'
                << solver_name()
                << "  Error occurred, outputting all data for diagnosis."
                << e.what();
      converged = false;
      output_mesh();
      output_data();
      output_vtk();
    }
}

template <int dim>
bool
StressSolver<dim>::has_converged() const
{
  return converged;
}

template <int dim>
const Triangulation<dim> &
StressSolver<dim>::get_mesh() const
{
  return triangulation;
}

template <int dim>
Triangulation<dim> &
StressSolver<dim>::get_mesh()
{
  return triangulation;
}

template <int dim>
const Vector<double> &
StressSolver<dim>::get_temperature() const
{
  return temperature;
}

template <int dim>
Vector<double> &
StressSolver<dim>::get_temperature()
{
  return temperature;
}

template <int dim>
const BlockVector<double> &
StressSolver<dim>::get_displacement() const
{
  return displacement;
}

template <int dim>
BlockVector<double> &
StressSolver<dim>::get_displacement()
{
  return displacement;
}

template <int dim>
const BlockVector<double> &
StressSolver<dim>::get_stress() const
{
  return stress;
}

template <int dim>
const BlockVector<double> &
StressSolver<dim>::get_stress_deviator() const
{
  return stress_deviator;
}

template <int dim>
const Vector<double> &
StressSolver<dim>::get_stress_hydrostatic() const
{
  return stress_hydrostatic;
}

template <int dim>
const Vector<double> &
StressSolver<dim>::get_stress_J_2() const
{
  return stress_J_2;
}

template <int dim>
const BlockVector<double> &
StressSolver<dim>::get_strain_e() const
{
  return strain_e;
}

template <int dim>
const BlockVector<double> &
StressSolver<dim>::get_strain_c() const
{
  return strain_c;
}

template <int dim>
BlockVector<double> &
StressSolver<dim>::get_strain_c()
{
  return strain_c;
}

template <int dim>
const ParameterHandler &
StressSolver<dim>::get_parameters() const
{
  return prm;
}

template <int dim>
ParameterHandler &
StressSolver<dim>::get_parameters()
{
  return prm;
}

template <int dim>
void
StressSolver<dim>::initialize()
{
  Timer timer;

  std::cout << solver_name() << "  Initializing finite element solution";

  dh_temp.distribute_dofs(fe_temp);
  dh.distribute_dofs(fe);

  const unsigned int n_dofs_temp = dh_temp.n_dofs();
  temperature.reinit(n_dofs_temp);
  displacement.reinit(dim, n_dofs_temp);
  stress.reinit(n_components, n_dofs_temp);
  stress_deviator.reinit(n_components, n_dofs_temp);
  strain_e.reinit(n_components, n_dofs_temp);
  strain_c.reinit(n_components, n_dofs_temp);
  stress_hydrostatic.reinit(n_dofs_temp);
  stress_von_Mises.reinit(n_dofs_temp);
  stress_J_2.reinit(n_dofs_temp);

  std::cout << " " << format_time(timer) << "\n";

  std::cout << solver_name() << "  "
            << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
            << solver_name() << "  "
            << "Number of degrees of freedom for temperature: " << n_dofs_temp
            << "\n";
}

template <int dim>
void
StressSolver<dim>::get_support_points(std::vector<Point<dim>> &points) const
{
  points.resize(dh_temp.n_dofs());
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dh_temp, points);
}

template <int dim>
void
StressSolver<dim>::set_bc1(const unsigned int id,
                           const unsigned int component,
                           const double       val)
{
  AssertThrow(component < dim,
              ExcMessage("Invalid component=" + std::to_string(component)));

  bc1_data[std::make_pair(id, component)] = val;
}

template <int dim>
void
StressSolver<dim>::set_bc_load(const unsigned int id, const Tensor<1, dim> val)
{
  bc_load_data[id] = val;
}

template <int dim>
unsigned int
StressSolver<dim>::get_degree() const
{
  return fe_temp.degree;
}

template <int dim>
const DoFHandler<dim> &
StressSolver<dim>::get_dof_handler() const
{
  return dh_temp;
}

template <int dim>
void
StressSolver<dim>::load_data()
{
  Timer timer;

  const std::string s = output_name_suffix();

  read_data(get_temperature(), "temperature" + s);
  read_data(get_displacement(), "displacement" + s);
  read_data(get_strain_c(), "strain_c" + s);
  // skip calculated quantities (stresses)

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
StressSolver<dim>::output_data() const
{
  Timer timer;

  const std::string s = output_name_suffix();

  write_data(get_temperature(), "temperature" + s);
  write_data(get_displacement(), "displacement" + s);
  write_data(get_stress(), "stress" + s);
  write_data(get_stress_deviator(), "stress_deviator" + s);
  write_data(get_stress_hydrostatic(), "stress_hydrostatic" + s);
  write_data(get_stress_J_2(), "stress_J_2" + s);
  write_data(get_strain_e(), "strain_e" + s);
  write_data(get_strain_c(), "strain_c" + s);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
StressSolver<dim>::output_vtk() const
{
  Timer timer;

  const std::string file_name = "result-stress" + output_name_suffix() + ".vtk";
  std::cout << solver_name() << "  Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh_temp);
  data_out.add_data_vector(temperature, "T");

  Vector<double> E(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    E[i] = calc_E(temperature[i]);
  data_out.add_data_vector(E, "E");

  Vector<double> C_11(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    C_11[i] = calc_C_11(temperature[i]);
  data_out.add_data_vector(C_11, "C_11");

  Vector<double> C_12(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    C_12[i] = calc_C_12(temperature[i]);
  data_out.add_data_vector(C_12, "C_12");

  Vector<double> C_44(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    C_44[i] = calc_C_44(temperature[i]);
  data_out.add_data_vector(C_44, "C_44");

  Vector<double> H(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    H[i] = calc_H(temperature[i]);
  data_out.add_data_vector(H, "H");

  Vector<double> alpha(temperature.size());
  for (unsigned int i = 0; i < temperature.size(); ++i)
    alpha[i] = calc_alpha(temperature[i]);
  data_out.add_data_vector(alpha, "alpha");

  for (unsigned int i = 0; i < displacement.n_blocks(); ++i)
    {
      const std::string name = "displacement_" + std::to_string(i);
      data_out.add_data_vector(displacement.block(i), name);
    }

  for (unsigned int i = 0; i < stress.n_blocks(); ++i)
    {
      const std::string name = "stress_" + std::to_string(i);
      data_out.add_data_vector(stress.block(i), name);
    }
  for (unsigned int i = 0; i < stress_deviator.n_blocks(); ++i)
    {
      const std::string name = "stress_deviator_" + std::to_string(i);
      data_out.add_data_vector(stress_deviator.block(i), name);
    }

  for (unsigned int i = 0; i < strain_e.n_blocks(); ++i)
    {
      const std::string name = "epsilon_e_" + std::to_string(i);
      data_out.add_data_vector(strain_e.block(i), name);
    }
  for (unsigned int i = 0; i < strain_c.n_blocks(); ++i)
    {
      const std::string name = "epsilon_c_" + std::to_string(i);
      data_out.add_data_vector(strain_c.block(i), name);
    }

  data_out.add_data_vector(stress_hydrostatic, "stress_hydrostatic");
  data_out.add_data_vector(stress_von_Mises, "stress_von_Mises");
  data_out.add_data_vector(stress_J_2, "stress_J_2");

  data_out.build_patches(prm.get_integer("Output subdivisions"));

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  data_out.write_vtk(output);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
StressSolver<dim>::output_mesh() const
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
StressSolver<dim>::output_parameter_table(const double       T1,
                                          const double       T2,
                                          const unsigned int n) const
{
  const std::string fname = "stress-parameter-table.tsv";
  std::cout << solver_name() << "  Saving table '" << fname << "', T=" << T1
            << "-" << T2 << " K, n=" << n << '\n';

  std::ofstream output(fname);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  output << "T[K]\t"
         << "E[Pa]\t"
         << "C_11[Pa]\t"
         << "C_12[Pa]\t"
         << "C_44[Pa]\t"
         << "H[Pa]\t"
         << "alpha[K^-1]\n";

  for (unsigned int i = 0; i < n; ++i)
    {
      const double T = T1 + (T2 - T1) * i / (n - 1);

      output << T << '\t' << calc_E(T) << '\t' << calc_C_11(T) << '\t'
             << calc_C_12(T) << '\t' << calc_C_44(T) << '\t' << calc_H(T)
             << '\t' << calc_alpha(T) << '\n';
    }
}

template <int dim>
void
StressSolver<dim>::prepare_for_solve()
{
  const unsigned int n_dofs_temp = dh_temp.n_dofs();

  system_rhs.reinit(dim, n_dofs_temp);

  // check whether already initialized
  if (!sparsity_pattern.empty() && !system_matrix.empty())
    return;

  BlockDynamicSparsityPattern dsp(dim, dim);
  for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          dsp.block(i, j).reinit(n_dofs_temp, n_dofs_temp);
        }
    }
  dsp.collect_sizes();

  DoFRenumbering::component_wise(dh);
  DoFTools::make_sparsity_pattern(dh, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
StressSolver<dim>::AssemblyScratchData::AssemblyScratchData(
  const Quadrature<dim> &   quadrature,
  const Quadrature<dim - 1> face_quadrature,
  const FiniteElement<dim> &fe_temp,
  const FiniteElement<dim> &fe)
  : fe_values_temp(fe_temp, quadrature, update_values)
  , fe_values(fe,
              quadrature,
              update_quadrature_points | update_values | update_gradients |
                update_JxW_values)
  , fe_face_values(fe, face_quadrature, update_values | update_JxW_values)
  , T_q(quadrature.size())
  , epsilon_c_q(n_components, std::vector<double>(quadrature.size()))
{}

template <int dim>
StressSolver<dim>::AssemblyScratchData::AssemblyScratchData(
  const AssemblyScratchData &scratch_data)
  : fe_values_temp(scratch_data.fe_values_temp.get_fe(),
                   scratch_data.fe_values_temp.get_quadrature(),
                   scratch_data.fe_values_temp.get_update_flags())
  , fe_values(scratch_data.fe_values.get_fe(),
              scratch_data.fe_values.get_quadrature(),
              scratch_data.fe_values.get_update_flags())
  , fe_face_values(scratch_data.fe_face_values.get_fe(),
                   scratch_data.fe_face_values.get_quadrature(),
                   scratch_data.fe_face_values.get_update_flags())
  , T_q(scratch_data.T_q)
  , epsilon_c_q(scratch_data.epsilon_c_q)
{}

template <int dim>
void
StressSolver<dim>::assemble_system()
{
  Timer timer;

  std::cout << solver_name() << "  Assembling system";

  const QGauss<dim> quadrature(
    prm.get_integer("Number of cell quadrature points"));
  const QGauss<dim - 1> face_quadrature(
    prm.get_integer("Number of face quadrature points"));

  system_matrix = 0;
  system_rhs    = 0;

  WorkStream::run(IteratorPair(
                    IteratorTuple(dh_temp.begin_active(), dh.begin_active())),
                  IteratorPair(IteratorTuple(dh_temp.end(), dh.end())),
                  *this,
                  &StressSolver::local_assemble_system,
                  &StressSolver::copy_local_to_global,
                  AssemblyScratchData(quadrature, face_quadrature, fe_temp, fe),
                  AssemblyCopyData());

  // Apply boundary conditions for displacement. Also check if BC was applied
  // for all displacement components; if not, set displacement=0 at single DOF.
  std::vector<bool> bc1_applied(dim, false);

  std::map<types::global_dof_index, double> boundary_values;
  for (auto const &it : bc1_data)
    {
      std::vector<bool> mask(dim, false);
      mask[it.first.second]        = true;
      bc1_applied[it.first.second] = true;

      VectorTools::interpolate_boundary_values( // break line
        dh,
        it.first.first,
        ConstantFunction<dim>(it.second, dim),
        boundary_values,
        mask);
    }
  for (unsigned int i = 0; i < bc1_applied.size(); ++i)
    {
      if (!bc1_applied[i])
        {
          boundary_values[i * temperature.size()] = 0;
#ifdef DEBUG
          std::cout << '\n'
                    << solver_name()
                    << "  displacement=0 applied for dimension " << i;
#endif
        }
    }

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     displacement,
                                     system_rhs);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
StressSolver<dim>::local_assemble_system(const IteratorPair & cell_pair,
                                         AssemblyScratchData &scratch_data,
                                         AssemblyCopyData &   copy_data)
{
  FEValues<dim> &            fe_values_temp  = scratch_data.fe_values_temp;
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

  std::vector<double> &             T_q         = scratch_data.T_q;
  std::vector<std::vector<double>> &epsilon_c_q = scratch_data.epsilon_c_q;

  std::vector<types::global_dof_index> &local_dof_indices =
    copy_data.local_dof_indices;

  local_dof_indices.resize(dofs_per_cell);

  std::vector<Tensor<1, n_components>> strains_ij(dofs_per_cell);

  Tensor<1, n_components> epsilon_T_q;

  const typename DoFHandler<dim>::active_cell_iterator &cell_temp =
    std::get<0>(*cell_pair);
  const typename DoFHandler<dim>::active_cell_iterator &cell =
    std::get<1>(*cell_pair);

  fe_values_temp.reinit(cell_temp);
  fe_values.reinit(cell);

  fe_values_temp.get_function_values(temperature, T_q);

  for (unsigned int i = 0; i < n_components; ++i)
    {
      fe_values_temp.get_function_values(strain_c.block(i), epsilon_c_q[i]);
    }

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const SymmetricTensor<2, n_components> stiffness =
        get_stiffness_tensor(T_q[q]);

      get_strain(T_q[q], epsilon_T_q);

      // precalculate
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          get_strain(fe_values, k, q, strains_ij[k]);
        }

      const double weight =
        dim == 2 ? fe_values.JxW(q) * fe_values.quadrature_point(q)[0] :
                   fe_values.JxW(q);

      // sum of thermal and creep strain
      Tensor<1, n_components> epsilon_T_c_q = epsilon_T_q;
      for (unsigned int i = 0; i < n_components; ++i)
        epsilon_T_c_q[i] += epsilon_c_q[i][q];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, n_components> strain_i_stiffness =
            strains_ij[i] * stiffness;

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_matrix(i, j) +=
                (strain_i_stiffness * strains_ij[j]) * weight;
            }
          cell_rhs(i) += (strain_i_stiffness * epsilon_T_c_q) * weight;
        }
    }

  for (unsigned int face_number = 0;
       face_number < GeometryInfo<dim>::faces_per_cell;
       ++face_number)
    {
      if (!cell->face(face_number)->at_boundary())
        continue;

      const auto it = bc_load_data.find(cell->face(face_number)->boundary_id());
      if (it == bc_load_data.end())
        continue;

      const Tensor<1, dim> F = it->second;

      fe_face_values.reinit(cell, face_number);

      for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          const double weight =
            dim == 2 ?
              fe_face_values.JxW(q) * fe_face_values.quadrature_point(q)[0] :
              fe_face_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;
              cell_rhs(i) +=
                F[component_i] * fe_face_values.shape_value(i, q) * weight;
            }
        }
    }

  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void
StressSolver<dim>::copy_local_to_global(const AssemblyCopyData &copy_data)
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
StressSolver<dim>::solve_system()
{
  Timer timer;

  std::cout << solver_name() << "  Solving system";

  const std::string solver_type = prm.get("Linear solver type");

  if (solver_type == "UMFPACK")
    {
      std::cout << " (" << solver_type << ")";

      SparseDirectUMFPACK A;
      A.initialize(system_matrix);
      A.vmult(displacement, system_rhs);
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

      SolverSelector<BlockVector<double>> solver;
      solver.select(solver_type);
      solver.set_control(control);

      // PreconditionSelector doesn't work with BlockSparseMatrix
      const std::string preconditioner_type = prm.get("Preconditioner type");
      if (preconditioner_type == "jacobi")
        {
          const double preconditioner_relaxation =
            prm.get_double("Preconditioner relaxation");

          PreconditionJacobi<BlockSparseMatrix<double>> preconditioner;
          preconditioner.initialize(system_matrix, preconditioner_relaxation);

          solver.solve(system_matrix, displacement, system_rhs, preconditioner);
        }
      else
        {
          PreconditionIdentity preconditioner;

          solver.solve(system_matrix, displacement, system_rhs, preconditioner);
        }

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

template <int dim>
double
StressSolver<dim>::calc_E(const double T) const
{
  return m_E.value(Point<1>(T));
}

template <int dim>
double
StressSolver<dim>::calc_C_11(const double T) const
{
  if (Cij_type == ElasticMatrixType::Cij)
    return m_C_11.value(Point<1>(T));

  return calc_E(T) * (1 - m_nu) / ((1 + m_nu) * (1 - 2 * m_nu));
}

template <int dim>
double
StressSolver<dim>::calc_C_12(const double T) const
{
  if (Cij_type == ElasticMatrixType::Cij)
    return m_C_12.value(Point<1>(T));

  return calc_E(T) * m_nu / ((1 + m_nu) * (1 - 2 * m_nu));
}

template <int dim>
double
StressSolver<dim>::calc_C_44(const double T) const
{
  if (Cij_type == ElasticMatrixType::Cij)
    return m_C_44.value(Point<1>(T));

  return calc_E(T) / (2 * (1 + m_nu));
}

template <int dim>
double
StressSolver<dim>::calc_H(const double T) const
{
  if (Cij_type == ElasticMatrixType::Cij)
    return 2 * calc_C_44(T) + calc_C_12(T) - calc_C_11(T);

  return 0;
}

template <int dim>
double
StressSolver<dim>::calc_alpha(const double T) const
{
  return m_alpha.value(Point<1>(T));
}

template <int dim>
void
StressSolver<dim>::recover_strain_extrapolation()
{
  prm.enter_subsection("Stress recovery");
  const QGauss<dim> quadrature(
    prm.get_integer("Number of cell quadrature points"));
  prm.leave_subsection();

  FEValues<dim> fe_values_temp(fe_temp, quadrature, update_default);
  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_quadrature_points | update_values |
                            update_gradients);

  const unsigned int n_dofs_temp        = dh_temp.n_dofs();
  const unsigned int dofs_per_cell_temp = fe_temp.dofs_per_cell;
  const unsigned int n_q_points         = quadrature.size();

  strain_e.reinit(n_components, n_dofs_temp);

  std::vector<unsigned int> count(n_dofs_temp, 0);

  FullMatrix<double> qpoint_to_dof_matrix(dofs_per_cell_temp, n_q_points);
  FETools::compute_projection_from_quadrature_points_matrix(
    fe_temp, quadrature, quadrature, qpoint_to_dof_matrix);

  // recover elastic strains
  std::vector<Vector<double>> displacement_q(n_q_points, Vector<double>(dim));

  std::vector<std::vector<Tensor<1, dim>>> grad_displacement_q(
    n_q_points, std::vector<Tensor<1, dim>>(dim));

  std::vector<Vector<double>> strain_q(n_components,
                                       Vector<double>(n_q_points));

  std::vector<Vector<double>> strain_cell(n_components,
                                          Vector<double>(dofs_per_cell_temp));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_temp);

  Tensor<1, n_components> epsilon_e_q;

  typename DoFHandler<dim>::active_cell_iterator cell_temp =
                                                   dh_temp.begin_active(),
                                                 cell = dh.begin_active(),
                                                 endc = dh.end();
  for (; cell != endc; ++cell_temp, ++cell)
    {
      fe_values_temp.reinit(cell_temp);
      fe_values.reinit(cell);

      fe_values.get_function_values(displacement, displacement_q);

      fe_values.get_function_gradients(displacement, grad_displacement_q);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          get_strain(fe_values.quadrature_point(q),
                     displacement_q[q],
                     grad_displacement_q[q],
                     epsilon_e_q);

          for (unsigned int k = 0; k < n_components; ++k)
            {
              strain_q[k][q] = epsilon_e_q[k];
            }
        }

      for (unsigned int k = 0; k < n_components; ++k)
        {
          qpoint_to_dof_matrix.vmult(strain_cell[k], strain_q[k]);
        }

      cell_temp->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell_temp; ++i)
        {
          count[local_dof_indices[i]] += 1;

          for (unsigned int k = 0; k < n_components; ++k)
            {
              strain_e.block(k)[local_dof_indices[i]] += strain_cell[k][i];
            }
        }
    }

  for (unsigned int k = 0; k < n_components; ++k)
    {
      for (unsigned int i = 0; i < n_dofs_temp; ++i)
        {
          AssertThrow(count[i] > 0,
                      ExcMessage("count[" + std::to_string(i) +
                                 "]=" + std::to_string(count[i]) +
                                 ", positive value expected"));

          strain_e.block(k)[i] /= count[i];
        }
    }
}

template <int dim>
void
StressSolver<dim>::recover_strain_global()
{
  prm.enter_subsection("Stress recovery");
  const QGauss<dim> quadrature(
    prm.get_integer("Number of cell quadrature points"));
  prm.leave_subsection();

  FEValues<dim> fe_values_temp(fe_temp, quadrature, update_values);
  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_quadrature_points | update_values |
                            update_gradients | update_JxW_values);

  const unsigned int n_dofs_temp        = dh_temp.n_dofs();
  const unsigned int dofs_per_cell_temp = fe_temp.dofs_per_cell;
  const unsigned int n_q_points         = quadrature.size();

  strain_e.reinit(n_components, n_dofs_temp, true);

  // prepare the global mass matrix and RHS
  DynamicSparsityPattern dsp(n_dofs_temp);
  DoFTools::make_sparsity_pattern(dh_temp, dsp);

  SparsityPattern sp;
  sp.copy_from(dsp);

  SparseMatrix<double> global_matrix;
  BlockVector<double>  global_rhs(n_components, n_dofs_temp);
  global_matrix.reinit(sp);

  FullMatrix<double>  cell_matrix(dofs_per_cell_temp, dofs_per_cell_temp);
  BlockVector<double> cell_rhs(n_components, dofs_per_cell_temp);

  // assemble the system for elastic strains
  std::vector<Vector<double>> displacement_q(n_q_points, Vector<double>(dim));

  std::vector<std::vector<Tensor<1, dim>>> grad_displacement_q(
    n_q_points, std::vector<Tensor<1, dim>>(dim));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_temp);

  Tensor<1, n_components> epsilon_e_q;

  typename DoFHandler<dim>::active_cell_iterator cell_temp =
                                                   dh_temp.begin_active(),
                                                 cell = dh.begin_active(),
                                                 endc = dh.end();
  for (; cell != endc; ++cell_temp, ++cell)
    {
      cell_matrix.reinit(dofs_per_cell_temp, dofs_per_cell_temp);
      cell_rhs.reinit(n_components, dofs_per_cell_temp);

      fe_values_temp.reinit(cell_temp);
      fe_values.reinit(cell);

      fe_values.get_function_values(displacement, displacement_q);

      fe_values.get_function_gradients(displacement, grad_displacement_q);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          get_strain(fe_values.quadrature_point(q),
                     displacement_q[q],
                     grad_displacement_q[q],
                     epsilon_e_q);

          const double weight =
            dim == 2 ? fe_values.JxW(q) * fe_values.quadrature_point(q)[0] :
                       fe_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell_temp; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell_temp; ++j)
                {
                  cell_matrix(i, j) += fe_values_temp.shape_value(i, q) *
                                       fe_values_temp.shape_value(j, q) *
                                       weight;
                }

              for (unsigned int k = 0; k < n_components; ++k)
                cell_rhs.block(k)(i) +=
                  fe_values_temp.shape_value(i, q) * epsilon_e_q[k] * weight;
            }
        }

      cell_temp->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        {
          for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
            global_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          for (unsigned int k = 0; k < n_components; ++k)
            global_rhs.block(k)(local_dof_indices[i]) += cell_rhs.block(k)(i);
        }
    }

  // solve the linear systems
  prm.enter_subsection("Stress recovery");
  const std::string solver_type = prm.get("Linear solver type");
  prm.leave_subsection();

  if (solver_type == "UMFPACK")
    {
      SparseDirectUMFPACK A;
      A.initialize(global_matrix);

      for (unsigned int k = 0; k < n_components; ++k)
        A.vmult(strain_e.block(k), global_rhs.block(k));
    }
  else
    {
      prm.enter_subsection("Stress recovery");
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
      prm.leave_subsection();

      PreconditionSelector<> preconditioner(preconditioner_type,
                                            preconditioner_relaxation);
      preconditioner.use_matrix(global_matrix);

      for (unsigned int k = 0; k < n_components; ++k)
        solver.solve(global_matrix,
                     strain_e.block(k),
                     global_rhs.block(k),
                     preconditioner);
    }
}

template <int dim>
void
StressSolver<dim>::calculate_stress_from_strain()
{
  const unsigned int n_dofs_temp = dh_temp.n_dofs();

  stress.reinit(n_components, n_dofs_temp, true);

  Tensor<1, n_components> epsilon_i, epsilon_T_i;

  for (unsigned int i = 0; i < n_dofs_temp; ++i)
    {
      const SymmetricTensor<2, n_components> stiffness =
        get_stiffness_tensor(temperature[i]);

      get_strain(temperature[i], epsilon_T_i);

      for (unsigned int k = 0; k < n_components; ++k)
        {
          epsilon_i[k] =
            strain_e.block(k)[i] - epsilon_T_i[k] - strain_c.block(k)[i];
        }

      const Tensor<1, n_components> s = stiffness * epsilon_i;

      for (unsigned int k = 0; k < n_components; ++k)
        {
          stress.block(k)[i] = s[k];
        }
    }
}

template <int dim>
void
StressSolver<dim>::calculate_stress(const bool skip_recovery)
{
  Timer timer;

  if (!skip_recovery)
    {
      prm.enter_subsection("Stress recovery");
      const std::string method = prm.get("Method");
      prm.leave_subsection();

      std::cout << solver_name() << "  Postprocessing results (" << method
                << ")";

      if (method == "extrapolation")
        recover_strain_extrapolation();
      else if (method == "global")
        recover_strain_global();
      else
        AssertThrow(false,
                    ExcMessage("calculate_stress: stress recovery method '" +
                               method + "' not supported."));
    }
  else
    std::cout << solver_name() << "  Postprocessing results (w/o recovery)";

  calculate_stress_from_strain();
  calculate_stress_invariants();

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
StressSolver<dim>::calculate_stress_invariants()
{
  const unsigned int n_dofs_temp = dh_temp.n_dofs();

  stress_hydrostatic.reinit(n_dofs_temp, true);
  stress_von_Mises.reinit(n_dofs_temp, true);
  stress_J_2.reinit(n_dofs_temp, true);

  for (unsigned int i = 0; i < n_dofs_temp; ++i)
    {
      stress_hydrostatic[i] =
        (stress.block(0)[i] + stress.block(1)[i] + stress.block(2)[i]) / 3;

      stress_von_Mises[i] = sqr(stress.block(0)[i] - stress.block(1)[i]) +
                            sqr(stress.block(1)[i] - stress.block(2)[i]) +
                            sqr(stress.block(2)[i] - stress.block(0)[i]);

      for (unsigned int k = 3; k < n_components; ++k)
        stress_von_Mises[i] += 6 * sqr(stress.block(k)[i]);

      stress_von_Mises[i] = std::sqrt(stress_von_Mises[i] / 2);

      stress_J_2[i] = sqr(stress_von_Mises[i]) / 3;
    }

  stress_deviator = stress;
  stress_deviator.block(0) -= stress_hydrostatic;
  stress_deviator.block(1) -= stress_hydrostatic;
  stress_deviator.block(2) -= stress_hydrostatic;
}

template <int dim>
SymmetricTensor<2, StressSolver<dim>::n_components>
StressSolver<dim>::get_stiffness_tensor(const double &T) const
{
  SymmetricTensor<2, n_components> tmp;

  if (Cij_type == ElasticMatrixType::Full)
    {
      const auto x = Point<1>(T);

      for (unsigned int j = 0; j < n_components; ++j)
        {
          for (unsigned int i = 0; i <= j; ++i)
            {
              tmp[i][j] = m_C_full[i + j * n_components].value(x);
            }
        }

      return tmp;
    }


  tmp[0][0] = tmp[1][1] = tmp[2][2] = calc_C_11(T);
  tmp[2][1] = tmp[2][0] = tmp[1][0] = calc_C_12(T);

  if (dim == 2)
    tmp[3][3] = calc_C_44(T);
  else if (dim == 3)
    tmp[3][3] = tmp[4][4] = tmp[5][5] = calc_C_44(T);

  return tmp;
}

template <int dim>
void
StressSolver<dim>::get_strain(const FEValues<dim> &    fe_values,
                              const unsigned int &     shape_func,
                              const unsigned int &     q,
                              Tensor<1, n_components> &strain) const
{
  if (dim == 2)
    {
      const auto grad_0 = fe_values.shape_grad_component(shape_func, q, 0);
      const auto grad_1 = fe_values.shape_grad_component(shape_func, q, 1);

      strain[0] = grad_0[0];
      strain[1] = grad_1[1];

      strain[2] = fe_values.shape_value_component(shape_func, q, 0) /
                  fe_values.quadrature_point(q)[0];

      strain[3] = grad_1[0] + grad_0[1];
    }
  else if (dim == 3)
    {
      const auto grad_0 = fe_values.shape_grad_component(shape_func, q, 0);
      const auto grad_1 = fe_values.shape_grad_component(shape_func, q, 1);
      const auto grad_2 = fe_values.shape_grad_component(shape_func, q, 2);

      strain[0] = grad_0[0];
      strain[1] = grad_1[1];
      strain[2] = grad_2[2];

      strain[3] = grad_2[1] + grad_1[2];
      strain[4] = grad_2[0] + grad_0[2];
      strain[5] = grad_1[0] + grad_0[1];
    }
}

template <int dim>
void
StressSolver<dim>::get_strain(const double &           T,
                              Tensor<1, n_components> &strain) const
{
  const double alpha_T = calc_alpha(T);

  strain[0] = strain[1] = strain[2] = alpha_T * (T - m_T_ref);

  if (dim == 2)
    strain[3] = 0;
  else if (dim == 3)
    strain[3] = strain[4] = strain[5] = 0;
}

template <int dim>
void
StressSolver<dim>::get_strain(
  const Point<dim> &                 point_q,
  const Vector<double> &             displacement_q,
  const std::vector<Tensor<1, dim>> &grad_displacement,
  Tensor<1, n_components> &          strain) const
{
  if (dim == 2)
    {
      strain[0] = grad_displacement[0][0];
      strain[1] = grad_displacement[1][1];

      strain[2] = displacement_q[0] / point_q[0];

      strain[3] = grad_displacement[1][0] + grad_displacement[0][1];
    }
  else if (dim == 3)
    {
      strain[0] = grad_displacement[0][0];
      strain[1] = grad_displacement[1][1];
      strain[2] = grad_displacement[2][2];

      strain[3] = grad_displacement[2][1] + grad_displacement[1][2];
      strain[4] = grad_displacement[2][0] + grad_displacement[0][2];
      strain[5] = grad_displacement[1][0] + grad_displacement[0][1];
    }
}

template <int dim>
std::string
StressSolver<dim>::output_name_suffix() const
{
  std::stringstream ss;
  ss << "-" << dim << "d-order" << get_degree();
  return ss.str();
}

template <int dim>
const std::vector<std::string>
StressSolver<dim>::stress_component_names() const
{
  if (dim == 2)
    return {"rr", "zz", "ff", "rz"};
  if (dim == 3)
    return {"xx", "yy", "zz", "yz", "xz", "xy"};

  AssertThrow(false, ExcNotImplemented());
  return {};
}

#endif
