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

/// Class for calculation of the thermal stresses for a given temperature field
template <int dim>
class StressSolver
{
public:
  /// Constructor

  /// Initializes the solver parameters from \c stress.prm.
  /// If it doesn't exist, the default parameter values are written to
  /// \c stress-default.prm.
  StressSolver(const unsigned int order = 2);

  /// Calculate the stresses
  void
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

  /// Initialize fields
  void
  initialize();

  /// Get coordinates of DOFs
  void
  get_support_points(std::vector<Point<dim>> &points) const;

  /// Get degrees of freedom for temperature
  const DoFHandler<dim> &
  get_dof_handler() const;

  /// Set first-type boundary condition
  void
  set_bc1(const unsigned int id,
          const unsigned int component,
          const double       val);

  /// Save results to disk
  void
  output_results() const;

  /// Save mesh to disk
  void
  output_mesh() const;

  /// Number of distinct elements of the stress tensor (3D: 6, 2D: 4)
  static const unsigned int n_components = 2 * dim;

private:
  /// Initialize parameters. Called by the constructor
  void
  initialize_parameters();

  /// Initialize data before calculation
  void
  prepare_for_solve();

  /// Assemble the system matrix and right-hand-side vector (multithreaded)
  void
  assemble_system();

  /// Structure that holds scratch data
  struct AssemblyScratchData
  {
    AssemblyScratchData(const Quadrature<dim> &   quadrature,
                        const FiniteElement<dim> &fe_temp,
                        const FiniteElement<dim> &fe);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);

    FEValues<dim> fe_values_temp;
    FEValues<dim> fe_values;

    std::vector<double>              T_q;
    std::vector<std::vector<double>> epsilon_c_q;
  };
  /// Structure that holds local contributions
  struct AssemblyCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };
  /// Iterator tuple
  typedef std_cxx11::tuple<typename DoFHandler<dim>::active_cell_iterator,
                           typename DoFHandler<dim>::active_cell_iterator>
    IteratorTuple;
  /// Iterator pair
  typedef SynchronousIterators<IteratorTuple> IteratorPair;
  /// Local assembly function
  void
  local_assemble_system(const IteratorPair & cell_pair,
                        AssemblyScratchData &scratch_data,
                        AssemblyCopyData &   copy_data);
  /// Copy local contributions to global
  void
  copy_local_to_global(const AssemblyCopyData &copy_data);

  /// Solve the system of linear equations
  void
  solve_system();

  /// Calculate the temperature-dependent Young's modulus
  double
  calc_E(const double T) const;

  /// Calculate the second-order elastic constant (stiffness) \f$C_{11}\f$, Pa
  double
  calc_C_11(const double T) const;

  /// Calculate the second-order elastic constant (stiffness) \f$C_{12}\f$, Pa
  double
  calc_C_12(const double T) const;

  /// Calculate the second-order elastic constant (stiffness) \f$C_{44}\f$, Pa
  double
  calc_C_44(const double T) const;

  /// Calculate the temperature-dependent thermal expansion coefficient
  double
  calc_alpha(const double T) const;

  /// Calculate the stresses from the displacement field
  void
  calculate_stress();

  /// Get stiffness tensor
  SymmetricTensor<2, StressSolver<dim>::n_components>
  get_stiffness_tensor(const double &T) const;

  /// Get strain from FEValues
  void
  get_strain(const FEValues<dim> &    fe_values,
             const unsigned int &     shape_func,
             const unsigned int &     q,
             Tensor<1, n_components> &strain) const;
  /// Get strain from temperature
  void
  get_strain(const double &T, Tensor<1, n_components> &strain) const;
  /// Get strain from displacement
  void
  get_strain(const std::vector<Tensor<1, dim>> &grad_displacement,
             Tensor<1, n_components> &          strain) const;

  Triangulation<dim> triangulation; ///< Mesh

  FE_Q<dim>       fe_temp;     ///< Finite element for temperature
  DoFHandler<dim> dh_temp;     ///< Degrees of freedom for temperature
  Vector<double>  temperature; ///< Temperature

  FESystem<dim>       fe;              ///< Finite element for displacement
  DoFHandler<dim>     dh;              ///< Degrees of freedom for displacement
  BlockVector<double> displacement;    ///< Displacement
  BlockVector<double> stress;          ///< Stress
  BlockVector<double> stress_deviator; ///< Stress deviator

  BlockVector<double> strain_c; ///< Creep strain

  Vector<double> stress_hydrostatic; ///< Mean (hydrostatic) stress
  Vector<double> stress_von_Mises;   ///< von Mises stress
  Vector<double> stress_J_2;         ///< Second invariant of deviatoric stress

  BlockSparsityPattern      sparsity_pattern; ///< Sparsity pattern
  BlockSparseMatrix<double> system_matrix;    ///< System matrix
  BlockVector<double>       system_rhs;       ///< Right-hand-side vector

  /// Data for first-type BC
  std::map<unsigned int, std::pair<unsigned int, double>> bc1_data;

  ParameterHandler prm; ///< Parameter handler

  ///< Young's modulus (temperature function), Pa
  FunctionParser<1> m_E;

  /// Thermal expansion coefficient (temperature function), K<sup>-1</sup>
  FunctionParser<1> m_alpha;

  double m_nu;    ///< Poisson's ratio, -
  double m_T_ref; ///< Reference temperature, K
};

template <int dim>
StressSolver<dim>::StressSolver(const unsigned int order)
  : fe_temp(order)
  , dh_temp(triangulation)
  , fe(FE_Q<dim>(order), dim)
  , dh(triangulation)
{
  std::cout << "Creating stress solver, order=" << order << ", dim=" << dim
            << " ("
#ifdef DEBUG
               "Debug"
#else
               "Release"
#endif
               ")\n";

  AssertThrow(dim == 3, ExcNotImplemented());

  const std::string info_T = " (accepts temperature function)";

  // Physical parameters from https://doi.org/10.1016/S0022-0248(01)01322-7
  prm.declare_entry("Young's modulus",
                    "1.56e11",
                    Patterns::Anything(),
                    "Young's modulus in Pa" + info_T);

  prm.declare_entry("Thermal expansion coefficient",
                    "3.2e-6",
                    Patterns::Anything(),
                    "Thermal expansion coefficient in 1/K" + info_T);

  prm.declare_entry("Poisson's ratio",
                    "0.25",
                    Patterns::Double(0, 0.5),
                    "Poisson's ratio (dimensionless)");

  prm.declare_entry("Reference temperature",
                    "1685",
                    Patterns::Double(),
                    "Reference temperature in K");

  prm.declare_entry("Linear solver type",
                    "minres",
                    Patterns::Selection("UMFPACK|" +
                                        SolverSelector<>::get_solver_names()),
                    "Name of linear solver");

  prm.declare_entry("Linear solver iterations",
                    "1000",
                    Patterns::Integer(0),
                    "Max number of iterations of linear solver");

  prm.declare_entry("Linear solver tolerance",
                    "1e-8",
                    Patterns::Double(0),
                    "Tolerance (max residual norm) of linear solver");

  prm.declare_entry("Preconditioner type",
                    "jacobi",
                    Patterns::Selection("none|jacobi"), // limited options
                    "Name of preconditioner");

  prm.declare_entry("Preconditioner relaxation",
                    "1.0",
                    Patterns::Double(0),
                    "Relaxation factor of preconditioner");

  prm.declare_entry("Number of threads",
                    "0",
                    Patterns::Integer(0),
                    "Maximum number of threads to be used (0 - default)");

  prm.declare_entry("Output precision",
                    "8",
                    Patterns::Integer(1),
                    "Precision of double variables for output of field data");

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
StressSolver<dim>::initialize_parameters()
{
  std::cout << "Intializing parameters";

  deallog.depth_console(2);

  const std::string m_E_expression = prm.get("Young's modulus");
  m_E.initialize("T", m_E_expression, typename FunctionParser<1>::ConstMap());

  const std::string m_alpha_expression =
    prm.get("Thermal expansion coefficient");
  m_alpha.initialize("T",
                     m_alpha_expression,
                     typename FunctionParser<1>::ConstMap());

  m_nu    = prm.get_double("Poisson's ratio");
  m_T_ref = prm.get_double("Reference temperature");

  const auto n_threads = prm.get_integer("Number of threads");
  MultithreadInfo::set_thread_limit(n_threads > 0 ? n_threads :
                                                    MultithreadInfo::n_cores());

  std::cout << "  done\n";

  std::cout << "E=" << m_E_expression << "\n"
            << "alpha=" << m_alpha_expression << "\n"
            << "nu=" << m_nu << "\n"
            << "T_ref=" << m_T_ref << "\n";

  std::cout << "n_cores=" << MultithreadInfo::n_cores() << "\n"
            << "n_threads=" << MultithreadInfo::n_threads() << "\n";
}

template <int dim>
void
StressSolver<dim>::solve()
{
  prepare_for_solve();
  assemble_system();
  solve_system();
  calculate_stress();
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
StressSolver<dim>::get_stress_J_2() const
{
  return stress_J_2;
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
void
StressSolver<dim>::initialize()
{
  Timer timer;

  std::cout << "Initializing finite element solution";

  dh_temp.distribute_dofs(fe_temp);
  dh.distribute_dofs(fe);

  const unsigned int n_dofs_temp = dh_temp.n_dofs();
  temperature.reinit(n_dofs_temp);
  displacement.reinit(dim, n_dofs_temp);
  stress.reinit(n_components, n_dofs_temp);
  stress_deviator.reinit(n_components, n_dofs_temp);
  strain_c.reinit(n_components, n_dofs_temp);

  std::cout << " " << format_time(timer) << "\n";

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
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

  bc1_data[id] = std::make_pair(component, val);
}

template <int dim>
const DoFHandler<dim> &
StressSolver<dim>::get_dof_handler() const
{
  return dh_temp;
}

template <int dim>
void
StressSolver<dim>::output_results() const
{
  Timer timer;

  std::stringstream ss;
  ss << "result-" << dim << "d-order" << fe.degree << ".vtk";
  const std::string file_name = ss.str();
  std::cout << "Saving to '" << file_name << "'";

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

  data_out.add_data_vector(stress_hydrostatic, "stress_hydrostatic");
  data_out.add_data_vector(stress_von_Mises, "stress_von_Mises");
  data_out.add_data_vector(stress_J_2, "stress_J_2");

  data_out.build_patches(fe.degree);

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

  std::stringstream ss;
  ss << "mesh-" << dim << "d-order" << fe.degree << ".msh";
  const std::string file_name = ss.str();
  std::cout << "Saving to '" << file_name << "'";

  std::ofstream output(file_name);

  GridOut grid_out;
  grid_out.set_flags(GridOutFlags::Msh(true));
  grid_out.write_msh(triangulation, output);

  std::cout << " " << format_time(timer) << "\n";
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
  const FiniteElement<dim> &fe_temp,
  const FiniteElement<dim> &fe)
  : fe_values_temp(fe_temp, quadrature, update_values)
  , fe_values(fe, quadrature, update_gradients | update_JxW_values)
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
  , T_q(scratch_data.T_q)
  , epsilon_c_q(scratch_data.epsilon_c_q)
{}

template <int dim>
void
StressSolver<dim>::assemble_system()
{
  AssertThrow(dim == 3, ExcNotImplemented());

  Timer timer;

  std::cout << "Assembling system";

  const QGauss<dim> quadrature(fe.degree + 1);

  system_matrix = 0;
  system_rhs    = 0;

  WorkStream::run(IteratorPair(
                    IteratorTuple(dh_temp.begin_active(), dh.begin_active())),
                  IteratorPair(IteratorTuple(dh_temp.end(), dh.end())),
                  *this,
                  &StressSolver::local_assemble_system,
                  &StressSolver::copy_local_to_global,
                  AssemblyScratchData(quadrature, fe_temp, fe),
                  AssemblyCopyData());

  // Apply boundary conditions for displacement
  std::map<types::global_dof_index, double> boundary_values;
  for (auto const &it : bc1_data)
    {
      std::vector<bool> mask(dim, false);
      mask[it.second.first] = true;

      VectorTools::interpolate_boundary_values( // break line
        dh,
        it.first,
        ConstantFunction<dim>(it.second.second, dim),
        boundary_values,
        mask);
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
  FEValues<dim> &        fe_values_temp = scratch_data.fe_values_temp;
  FEValues<dim> &        fe_values      = scratch_data.fe_values;
  const Quadrature<dim> &quadrature     = fe_values.get_quadrature();

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature.size();

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

  Tensor<1, n_components> strain_i, strain_j, epsilon_T_q;

  const typename DoFHandler<dim>::active_cell_iterator &cell_temp =
    std_cxx11::get<0>(*cell_pair);
  const typename DoFHandler<dim>::active_cell_iterator &cell =
    std_cxx11::get<1>(*cell_pair);

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

      // sum of thermal and creep strain
      Tensor<1, n_components> epsilon_T_c_q = epsilon_T_q;
      for (unsigned int i = 0; i < n_components; ++i)
        epsilon_T_c_q[i] += epsilon_c_q[i][q];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          get_strain(fe_values, i, q, strain_i);

          const Tensor<1, n_components> strain_i_stiffness =
            strain_i * stiffness;

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              get_strain(fe_values, j, q, strain_j);

              cell_matrix(i, j) +=
                (strain_i_stiffness * strain_j) * fe_values.JxW(q);
            }
          cell_rhs(i) +=
            (strain_i_stiffness * epsilon_T_c_q) * fe_values.JxW(q);
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

  std::cout << "Solving system";

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
      std::cout << "\n";

      const int solver_iterations = prm.get_integer("Linear solver iterations");
      const double solver_tolerance = prm.get_double("Linear solver tolerance");

      const bool log_history = false;
      const bool log_result =
#ifdef DEBUG
        true
#else
        false
#endif
        ;

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
  return calc_E(T) * (1 - m_nu) / ((1 + m_nu) * (1 - 2 * m_nu));
}

template <int dim>
double
StressSolver<dim>::calc_C_12(const double T) const
{
  return calc_E(T) * m_nu / ((1 + m_nu) * (1 - 2 * m_nu));
}

template <int dim>
double
StressSolver<dim>::calc_C_44(const double T) const
{
  return calc_E(T) / (2 * (1 + m_nu));
}

template <int dim>
double
StressSolver<dim>::calc_alpha(const double T) const
{
  return m_alpha.value(Point<1>(T));
}

template <int dim>
void
StressSolver<dim>::calculate_stress()
{
  AssertThrow(dim == 3, ExcNotImplemented());

  Timer timer;

  std::cout << "Postprocessing results";

  const QGauss<dim> quadrature(fe.degree + 1);

  FEValues<dim> fe_values_temp(fe_temp, quadrature, update_values);
  FEValues<dim> fe_values(fe, quadrature, update_gradients);

  const unsigned int n_dofs_temp        = dh_temp.n_dofs();
  const unsigned int dofs_per_cell_temp = fe_temp.dofs_per_cell;
  const unsigned int n_q_points         = quadrature.size();

  stress.reinit(n_components, n_dofs_temp);
  stress_hydrostatic.reinit(n_dofs_temp);
  stress_von_Mises.reinit(n_dofs_temp);
  stress_J_2.reinit(n_dofs_temp);

  std::vector<unsigned int> count(n_dofs_temp, 0);

  FullMatrix<double> qpoint_to_dof_matrix(dofs_per_cell_temp, n_q_points);
  FETools::compute_projection_from_quadrature_points_matrix(
    fe_temp, quadrature, quadrature, qpoint_to_dof_matrix);

  std::vector<double> T_q(n_q_points);

  std::vector<std::vector<double>> epsilon_c_q(n_components,
                                               std::vector<double>(n_q_points));

  std::vector<std::vector<Tensor<1, dim>>> grad_displacement_q(
    n_q_points, std::vector<Tensor<1, dim>>(dim));

  std::vector<Vector<double>> stress_q(n_components,
                                       Vector<double>(n_q_points));

  std::vector<Vector<double>> stress_cell(n_components,
                                          Vector<double>(dofs_per_cell_temp));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_temp);

  Tensor<1, n_components> epsilon_T_q, epsilon_e_q;

  typename DoFHandler<dim>::active_cell_iterator cell_temp =
                                                   dh_temp.begin_active(),
                                                 cell = dh.begin_active(),
                                                 endc = dh.end();
  for (; cell != endc; ++cell_temp, ++cell)
    {
      fe_values_temp.reinit(cell_temp);
      fe_values.reinit(cell);

      fe_values_temp.get_function_values(temperature, T_q);

      fe_values.get_function_gradients(displacement, grad_displacement_q);

      for (unsigned int i = 0; i < n_components; ++i)
        fe_values_temp.get_function_values(strain_c.block(i), epsilon_c_q[i]);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const SymmetricTensor<2, n_components> stiffness =
            get_stiffness_tensor(T_q[q]);

          get_strain(T_q[q], epsilon_T_q);
          get_strain(grad_displacement_q[q], epsilon_e_q);

          // sum of thermal and creep strain
          Tensor<1, n_components> epsilon_T_c_q = epsilon_T_q;
          for (unsigned int i = 0; i < n_components; ++i)
            epsilon_T_c_q[i] += epsilon_c_q[i][q];

          const Tensor<1, n_components> s =
            stiffness * (epsilon_e_q - epsilon_T_c_q);

          for (unsigned int k = 0; k < n_components; ++k)
            {
              stress_q[k][q] = s[k];
            }
        }

      for (unsigned int k = 0; k < n_components; ++k)
        {
          qpoint_to_dof_matrix.vmult(stress_cell[k], stress_q[k]);
        }

      cell_temp->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell_temp; ++i)
        {
          count[local_dof_indices[i]] += 1;

          for (unsigned int k = 0; k < n_components; ++k)
            {
              stress.block(k)[local_dof_indices[i]] += stress_cell[k][i];
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

          stress.block(k)[i] /= count[i];
        }
    }

  for (unsigned int i = 0; i < n_dofs_temp; ++i)
    {
      stress_hydrostatic[i] =
        (stress.block(0)[i] + stress.block(1)[i] + stress.block(2)[i]) / 3;

      stress_von_Mises[i] =
        sqr(stress.block(0)[i] - stress.block(1)[i]) +
        sqr(stress.block(1)[i] - stress.block(2)[i]) +
        sqr(stress.block(2)[i] - stress.block(0)[i]) +
        6 * (sqr(stress.block(3)[i]) + sqr(stress.block(4)[i]) +
             sqr(stress.block(5)[i]));

      stress_von_Mises[i] = std::sqrt(stress_von_Mises[i] / 2);

      stress_J_2[i] = sqr(stress_von_Mises[i]) / 3;
    }

  stress_deviator = stress;
  stress_deviator.block(0) -= stress_hydrostatic;
  stress_deviator.block(1) -= stress_hydrostatic;
  stress_deviator.block(2) -= stress_hydrostatic;

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
SymmetricTensor<2, StressSolver<dim>::n_components>
StressSolver<dim>::get_stiffness_tensor(const double &T) const
{
  AssertThrow(dim == 3, ExcNotImplemented());

  SymmetricTensor<2, n_components> tmp;
  tmp[0][0] = tmp[1][1] = tmp[2][2] = calc_C_11(T);
  tmp[3][3] = tmp[4][4] = tmp[5][5] = calc_C_44(T);
  tmp[2][1] = tmp[2][0] = tmp[1][0] = calc_C_12(T);

  return tmp;
}

template <int dim>
void
StressSolver<dim>::get_strain(const FEValues<dim> &    fe_values,
                              const unsigned int &     shape_func,
                              const unsigned int &     q,
                              Tensor<1, n_components> &strain) const
{
  AssertThrow(dim == 3, ExcNotImplemented());

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

template <int dim>
void
StressSolver<dim>::get_strain(const double &           T,
                              Tensor<1, n_components> &strain) const
{
  AssertThrow(dim == 3, ExcNotImplemented());

  const double alpha_T = calc_alpha(T);

  strain[0] = strain[1] = strain[2] = alpha_T * (T - m_T_ref);
  strain[3] = strain[4] = strain[5] = 0;
}

template <int dim>
void
StressSolver<dim>::get_strain(
  const std::vector<Tensor<1, dim>> &grad_displacement,
  Tensor<1, n_components> &          strain) const
{
  AssertThrow(dim == 3, ExcNotImplemented());

  strain[0] = grad_displacement[0][0];
  strain[1] = grad_displacement[1][1];
  strain[2] = grad_displacement[2][2];

  strain[3] = grad_displacement[2][1] + grad_displacement[1][2];
  strain[4] = grad_displacement[2][0] + grad_displacement[0][2];
  strain[5] = grad_displacement[1][0] + grad_displacement[0][1];
}

#endif