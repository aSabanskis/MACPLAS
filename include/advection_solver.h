#ifndef macplas_advection_solver_h
#define macplas_advection_solver_h

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

#include "utilities.h"

using namespace dealii;

/** Class for applying pure advection to fields.
 * Solves \f$ \partial f / \partial t + \vec{u} \nabla f = 0\f$.
 */
template <int dim>
class AdvectionSolver
{
public:
  /** Constructor.
   * Initializes the solver parameters from \c advection.prm.
   * If it doesn't exist, the default parameter values are written to
   * \c advection-default.prm.
   * Default values are used and written to \c advection.prm if
   * \c use_default_prm parameter is specified.
   */
  AdvectionSolver(const unsigned int order           = 2,
                  const bool         use_default_prm = false);

  /** Solver name
   */
  std::string
  solver_name() const;

  /** Calculate new values of all fields.
   * @returns \c true if the final time has not been reached
   */
  bool
  solve();

  /** Set the velocity field
   */
  void
  set_velocity(const std::vector<Tensor<1, dim>> &u);

  /** Get mesh
   */
  const Triangulation<dim> &
  get_mesh() const;

  /** Get mesh
   */
  Triangulation<dim> &
  get_mesh();

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

  /** Get parameters AdvectionSolver::prm
   */
  const ParameterHandler &
  get_parameters() const;

  /** Get parameters AdvectionSolver::prm
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

  /** Get finite element degree
   */
  unsigned int
  get_degree() const;

  /** Set first-type boundary condition \f$f = f_0\f$.
   * The value does not need to be specified.
   */
  void
  set_bc1(const unsigned int id);

  /** Add a FE field
   */
  void
  add_field(const std::string &name, const Vector<double> &field);

  /** Get calculated gradients by field name
   */
  const Vector<double> &
  get_field(const std::string &name) const;

  /** Save results to disk in \c vtk format
   */
  void
  output_vtk() const;

  /** Save results at DOFs of boundary \c id to disk
   */
  void
  output_boundary_values(const unsigned int id) const;

private:
  /** Stabilization type
   */
  enum StabilizationType
  {
    supg, ///< Streamline-Upwind Petrov-Galerkin
    gls   ///< Galerkin Least-Squares
  };

  /** Initialize parameters. Called by the constructor
   */
  void
  initialize_parameters();

  /** Time stepping: advance time \f$t \to t + \Delta t\f$
   */
  void
  advance_time();

  /** Initialize data before calculation
   */
  void
  prepare_for_solve();

  /** Assemble the system matrix and right-hand-side vector
   */
  void
  assemble_system();

  /** Solve the system of linear equations
   */
  void
  solve_system();

  /** Helper method for creating output file name.
   * @returns \c "-<dim>d-order<order>-t<time>"
   */
  std::string
  output_name_suffix() const;

  /** Velocity field split into components and magnitude, m/s
   */
  BlockVector<double> velocity;

  /** All FE fields
   */
  std::map<std::string, Vector<double>> fields;

  /** All FE fields at the previous time step.
   * \c map is not used for convenience in \c solve.
   */
  BlockVector<double> fields_prev;

  /** Stabilization factor for all DoFs
   */
  Vector<double> stabilization_factor;

  /** Stabilization type
   */
  StabilizationType stabilization_type;

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
  BlockVector<double> system_rhs;

  /** Boundary IDs for first-type BC
   */
  std::set<unsigned int> bc1;

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
AdvectionSolver<dim>::AdvectionSolver(const unsigned int order,
                                      const bool         use_default_prm)
  : fe(order)
  , dh(triangulation)
  , current_time(0)
  , current_time_step(0)
{
  std::cout << "Creating advection solver"
            << " ("
#ifdef DEBUG
               "Debug"
#else
               "Release"
#endif
               ")\n";

  prm.declare_entry("Time step",
                    "1",
                    Patterns::Double(0),
                    "Time step in seconds (0 - steady-state)");

  prm.declare_entry("Max time",
                    "10",
                    Patterns::Double(0),
                    "Maximum time in seconds");

  prm.declare_entry("Linear solver type",
                    "fgmres",
                    Patterns::Selection("UMFPACK|" +
                                        SolverSelector<>::get_solver_names()),
                    "Name of linear solver");

  prm.declare_entry("Linear solver iterations",
                    "1000",
                    Patterns::Integer(0),
                    "Maximum number of iterations of linear solver");

  prm.declare_entry("Linear solver tolerance",
                    "1e-12",
                    Patterns::Double(0),
                    "Tolerance (maximum residual norm) of linear solver");

  prm.declare_entry(
    "Linear solver relative tolerance",
    "1e-2",
    Patterns::Double(0),
    "Relative tolerance (residual improvement) of linear solver");

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

  prm.declare_entry("Stabilization type",
                    "SUPG",
                    Patterns::Selection("SUPG|GLS"),
                    "Type of numerical stabilization");

  prm.declare_entry("Stabilization multiplier",
                    "1",
                    Patterns::Double(0),
                    "Multiplier for stabilization (0: no stabilization)");

  prm.declare_entry("Number of cell quadrature points",
                    "0",
                    Patterns::Integer(0),
                    "Number of QGauss<dim> quadrature points (0: order+1)");

  prm.declare_entry(
    "Output precision",
    "8",
    Patterns::Integer(1),
    "Precision of double variables for output of field and probe data");

  prm.declare_entry("Output subdivisions",
                    "0",
                    Patterns::Integer(0),
                    "Number of cell subdivisions for vtk output (0: order)");

  if (use_default_prm)
    {
      std::ofstream of("advection.prm");
      prm.print_parameters(of, ParameterHandler::Text);
    }
  else
    try
      {
        prm.parse_input("advection.prm");
      }
    catch (std::exception &e)
      {
        std::cout << e.what() << "\n";

        std::ofstream of("advection-default.prm");
        prm.print_parameters(of, ParameterHandler::Text);
      }

  initialize_parameters();
}

template <int dim>
void
AdvectionSolver<dim>::initialize_parameters()
{
  std::cout << solver_name() << "  Initializing parameters";

  get_time_step() = prm.get_double("Time step");

  const long int n_q_default = get_degree() + 1;

  if (prm.get_integer("Number of cell quadrature points") == 0)
    prm.set("Number of cell quadrature points", n_q_default);

  const long int n_vtk_default = get_degree();

  if (prm.get_integer("Output subdivisions") == 0)
    prm.set("Output subdivisions", n_vtk_default);

  const std::string st = prm.get("Stabilization type");
  if (st == "SUPG")
    stabilization_type = supg;
  else if (st == "GLS")
    stabilization_type = gls;
  else
    AssertThrow(false, ExcMessage("Unsupported stabilization type " + st));

  std::cout << "  done\n";

  std::cout << "n_q_cell=" << prm.get("Number of cell quadrature points")
            << "\n";
}

template <int dim>
std::string
AdvectionSolver<dim>::solver_name() const
{
  return "MACPLAS:Advection";
}

template <int dim>
const Triangulation<dim> &
AdvectionSolver<dim>::get_mesh() const
{
  return triangulation;
}

template <int dim>
Triangulation<dim> &
AdvectionSolver<dim>::get_mesh()
{
  return triangulation;
}

template <int dim>
double
AdvectionSolver<dim>::get_time() const
{
  return current_time;
}

template <int dim>
double &
AdvectionSolver<dim>::get_time()
{
  return current_time;
}

template <int dim>
double
AdvectionSolver<dim>::get_time_step() const
{
  return current_time_step;
}

template <int dim>
double &
AdvectionSolver<dim>::get_time_step()
{
  return current_time_step;
}

template <int dim>
double
AdvectionSolver<dim>::get_max_time() const
{
  return prm.get_double("Max time");
}

template <int dim>
const ParameterHandler &
AdvectionSolver<dim>::get_parameters() const
{
  return prm;
}

template <int dim>
ParameterHandler &
AdvectionSolver<dim>::get_parameters()
{
  return prm;
}

template <int dim>
void
AdvectionSolver<dim>::initialize()
{
  Timer timer;

  std::cout << solver_name() << "  Initializing finite element solution";

  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();

  velocity.reinit(dim + 1, n_dofs); // dim components and magnitude
  stabilization_factor.reinit(n_dofs);

  std::cout << " " << format_time(timer) << "\n";

  std::cout << solver_name() << "  "
            << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
            << solver_name() << "  "
            << "Number of degrees of freedom: " << n_dofs << "\n";
}

template <int dim>
void
AdvectionSolver<dim>::get_boundary_points(
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
AdvectionSolver<dim>::get_support_points(std::vector<Point<dim>> &points) const
{
  points.resize(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dh, points);
}

template <int dim>
unsigned int
AdvectionSolver<dim>::get_degree() const
{
  return fe.degree;
}

template <int dim>
void
AdvectionSolver<dim>::set_bc1(const unsigned int id)
{
  bc1.insert(id);
}

template <int dim>
void
AdvectionSolver<dim>::add_field(const std::string &   name,
                                const Vector<double> &field)
{
  fields[name] = field;
}

template <int dim>
bool
AdvectionSolver<dim>::solve()
{
  advance_time();

  const double t     = get_time();
  const double dt    = get_time_step();
  const double t_max = get_max_time();

  const unsigned int n_fields = fields.size();

  // allow negative times which would simply reverse the velocity
  if (dt == 0)
    return false;

  prepare_for_solve();
  assemble_system();
  solve_system();

  std::cout << solver_name() << "  "
            << "Time " << t << " s"
            << " step " << dt << " s"
            << " n_fields=" << n_fields << "\n";

  return t + 1e-4 * dt < t_max;
}

template <int dim>
void
AdvectionSolver<dim>::set_velocity(const std::vector<Tensor<1, dim>> &u)
{
  const unsigned int n_dofs = dh.n_dofs();

  AssertDimension(n_dofs, u.size());

  velocity.reinit(dim + 1, n_dofs, true); // dim components and magnitude
  for (unsigned int k = 0; k < dim; ++k)
    {
      auto &U = velocity.block(k);
      for (unsigned int i = 0; i < n_dofs; ++i)
        U[i] = u[i][k];
    }

  auto &U_mag = velocity.block(dim);
  for (unsigned int i = 0; i < n_dofs; ++i)
    U_mag[i] = u[i].norm();
}

template <int dim>
void
AdvectionSolver<dim>::advance_time()
{
  double &     dt     = get_time_step();
  const double t_prev = get_time();
  const double t_max  = get_max_time();

  // stop exactly at max time
  if (t_prev + (1 + 1e-4) * dt >= t_max)
    dt = t_max - t_prev;

  get_time() += get_time_step();
}

template <int dim>
void
AdvectionSolver<dim>::prepare_for_solve()
{
  const unsigned int n_dofs   = dh.n_dofs();
  const unsigned int n_fields = fields.size();

  system_rhs.reinit(n_fields, n_dofs);
  fields_prev.reinit(n_fields);
  stabilization_factor.reinit(n_dofs);

  unsigned int k = 0;
  for (const auto &it : fields)
    {
      AssertDimension(n_dofs, it.second.size());
      fields_prev.block(k) = it.second;
      ++k;
    }

  const double tau0 = prm.get_double("Stabilization multiplier");

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = dh.begin_active(),
                                                 endc = dh.end();
  if (tau0 > 0)
    for (; cell != endc; ++cell)
      {
        const double h = cell->diameter();

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int j = local_dof_indices[i];

            if (velocity.block(dim)[j] > 0)
              stabilization_factor[j] =
                std::max(stabilization_factor[j],
                         tau0 * h / (2 * velocity.block(dim)[j]));
          }
      }

  if (!sparsity_pattern.empty() && !system_matrix.empty())
    return;

  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_sparsity_pattern(dh, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void
AdvectionSolver<dim>::assemble_system()
{
  Timer timer;

  std::cout << solver_name() << "  Assembling system";

  const double dt     = get_time_step();
  const double tau_dt = stabilization_type == gls ? 1 / dt : 0;

  const QGauss<dim> quadrature(
    prm.get_integer("Number of cell quadrature points"));

  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int n_dofs        = dh.n_dofs();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature.size();
  const unsigned int n_fields      = fields_prev.n_blocks();

  std::vector<std::vector<double>> f_prev_q(n_fields,
                                            std::vector<double>(n_q_points));

  std::vector<double>         u_component_q(n_q_points);
  std::vector<Tensor<1, dim>> u_q(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FullMatrix<double>  cell_matrix(dofs_per_cell, dofs_per_cell);
  BlockVector<double> cell_rhs(n_fields, dofs_per_cell);

  system_matrix = 0;
  system_rhs    = 0;

  typename DoFHandler<dim>::active_cell_iterator cell = dh.begin_active(),
                                                 endc = dh.end();
  for (; cell != endc; ++cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(n_fields, dofs_per_cell);

      fe_values.reinit(cell);

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int k = 0; k < n_fields; ++k)
        fe_values.get_function_values(fields_prev.block(k), f_prev_q[k]);

      for (unsigned int k = 0; k < dim; ++k)
        {
          fe_values.get_function_values(velocity.block(k), u_component_q);
          for (unsigned int q = 0; q < n_q_points; ++q)
            u_q[q][k] = u_component_q[q];
        }

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const double weight = fe_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double tau    = stabilization_factor[local_dof_indices[i]];
              const double phi_i0 = fe_values.shape_value(i, q);
              const double phi_i =
                phi_i0 +
                tau * (phi_i0 * tau_dt + u_q[q] * fe_values.shape_grad(i, q));

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const double          phi_j = fe_values.shape_value(j, q);
                  const Tensor<1, dim> &grad_phi_j = fe_values.shape_grad(j, q);

                  cell_matrix(i, j) +=
                    (phi_j / dt + (u_q[q] * grad_phi_j)) * phi_i * weight;
                }

              for (unsigned int k = 0; k < n_fields; ++k)
                cell_rhs.block(k)(i) += f_prev_q[k][q] / dt * phi_i * weight;
            }
        }

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          for (unsigned int k = 0; k < n_fields; ++k)
            system_rhs.block(k)(local_dof_indices[i]) += cell_rhs.block(k)(i);
        }
    }

  // apply Dirichlet boundary conditions
  for (const auto &b : bc1)
    {
      std::vector<bool> boundary_dofs(n_dofs, false);

      DoFTools::extract_boundary_dofs(dh,
                                      ComponentMask(),
                                      boundary_dofs,
                                      {static_cast<types::boundary_id>(b)});

      for (unsigned int k = 0; k < n_fields; ++k)
        {
          std::map<types::global_dof_index, double> boundary_values;

          const Vector<double> &f = fields_prev.block(k);

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              if (boundary_dofs[i])
                boundary_values[i] = f[i];
            }

          // The system matrix is reused for all fields, therefore the column
          // elimination must be disabled.
          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             fields_prev.block(k),
                                             system_rhs.block(k),
                                             false);
        }
    }

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
AdvectionSolver<dim>::solve_system()
{
  Timer timer;

  std::cout << solver_name() << "  Solving system";

  const unsigned int n_fields = fields_prev.n_blocks();

  const std::string solver_type = prm.get("Linear solver type");

  if (solver_type == "UMFPACK")
    {
      std::cout << " (" << solver_type << ")";

      SparseDirectUMFPACK A;
      A.initialize(system_matrix);

      for (unsigned int k = 0; k < n_fields; ++k)
        {
          A.vmult(fields_prev.block(k), system_rhs.block(k));
        }
    }
  else
    {
      const unsigned int solver_iterations =
        prm.get_integer("Linear solver iterations");
      const double solver_rel_tolerance =
        prm.get_double("Linear solver relative tolerance");

      const bool log_history = prm.get_bool("Log convergence full");
      const bool log_result  = prm.get_bool("Log convergence final");

      if (log_history || log_result)
        std::cout << "\n";

      const std::string preconditioner_type = prm.get("Preconditioner type");
      const double      preconditioner_relaxation =
        prm.get_double("Preconditioner relaxation");

      PreconditionSelector<> preconditioner(preconditioner_type,
                                            preconditioner_relaxation);
      preconditioner.use_matrix(system_matrix);

      for (unsigned int k = 0; k < n_fields; ++k)
        {
          double solver_tolerance = prm.get_double("Linear solver tolerance");

          const double rhs_norm = system_rhs.block(k).l2_norm();

          if (rhs_norm > 0)
            solver_tolerance *= rhs_norm;

          ReductionControl control(solver_iterations,
                                   solver_tolerance,
                                   solver_rel_tolerance,
                                   log_history,
                                   log_result);
          SolverSelector<> solver;
          solver.select(solver_type);
          solver.set_control(control);

          try
            {
              solver.solve(system_matrix,
                           fields_prev.block(k),
                           system_rhs.block(k),
                           preconditioner);
            }
          catch (dealii::SolverControl::NoConvergence &e)
            {
              if (control.last_step() < solver_iterations)
                throw;
              // otherwise: maximum number of iterations reached, do nothing
            }

          if (control.last_step() >= solver_iterations ||
              (control.last_value() >= solver_tolerance &&
               control.last_value() >=
                 control.initial_value() * solver_rel_tolerance))
            {
              if (!(log_history || log_result))
                std::cout << "\n";

              std::cout << solver_name()
                        << "  Warning: not converged! Residual(0)="
                        << control.initial_value() << " Residual("
                        << control.last_step() << ")=" << control.last_value()
                        << " tol=" << solver_tolerance << "\n";
            }
        }
    }

  unsigned int k = 0;
  for (auto &it : fields)
    {
      it.second = fields_prev.block(k);
      ++k;
    }

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
std::string
AdvectionSolver<dim>::output_name_suffix() const
{
  std::stringstream ss;
  ss << std::setprecision(8);
  ss << "-" << dim << "d-order" << get_degree() << "-t" << get_time();
  return ss.str();
}

template <int dim>
const Vector<double> &
AdvectionSolver<dim>::get_field(const std::string &name) const
{
  return fields.at(name);
}

template <int dim>
void
AdvectionSolver<dim>::output_vtk() const
{
  Timer timer;

  const std::string file_name =
    "result-advection" + output_name_suffix() + ".vtk";
  std::cout << solver_name() << "  Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh);

  for (const auto &it : fields)
    data_out.add_data_vector(it.second, it.first);

  for (unsigned int k = 0; k < dim; ++k)
    data_out.add_data_vector(velocity.block(k), "u_" + std::to_string(k));

  data_out.add_data_vector(stabilization_factor, "stabilization_factor");

  data_out.build_patches(prm.get_integer("Output subdivisions"));

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  data_out.write_vtk(output);

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
AdvectionSolver<dim>::output_boundary_values(const unsigned int id) const
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

  const std::string file_name = "result-advection" + output_name_suffix() +
                                "-boundary" + std::to_string(id) + ".dat";
  std::cout << solver_name() << "  Saving to '" << file_name << "'";

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  const auto dims = coordinate_names(dim);
  for (const auto &d : dims)
    output << d << "[m]\t";

  for (const auto &it : fields)
    output << it.first << '\t';

  for (unsigned int k = 0; k < dim; ++k)
    output << "u_" + std::to_string(k) << '\t';

  output << "stabilization_factor";

  output << '\n';


  for (unsigned int i = 0; i < points.size(); ++i)
    {
      if (!boundary_dofs[i])
        continue;

      // a simple '<< points[i]' would put space between coordinates
      for (unsigned int d = 0; d < dim; ++d)
        output << points[i][d] << '\t';

      for (const auto &it : fields)
        output << it.second[i] << '\t';

      for (unsigned int k = 0; k < dim; ++k)
        output << velocity.block(k)[i] << '\t';

      output << stabilization_factor[i] << '\n';
    }

  std::cout << " " << format_time(timer) << "\n";
}

#endif