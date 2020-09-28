#ifndef macplas_temperature_solver_h
#define macplas_temperature_solver_h

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/polynomial.h>
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

using namespace dealii;

/// Stefan–Boltzmann constant, W m<sup>-2</sup> K<sup>-4</sup>
static const double sigma_SB = 5.67e-8;

/// Data structure for thermal radiation and incoming heat flux density BC
struct heat_flux_data
{
  /// Incoming heat flux density, W m<sup>-2</sup>
  Vector<double> q_in;
  /// Temperature-dependent emissivity, -
  std::function<double(double)> emissivity;
  /// Emissivity temperature derivative, K<sup>-1</sup>
  std::function<double(double)> emissivity_deriv;
};

/// Class for calculation of the time-dependent temperature field
template <int dim>
class TemperatureSolver
{
public:
  /// Constructor

  /// Initializes the solver parameters from \c temperature.prm.
  /// If it doesn't exist, the default parameter values are written to
  /// \c temperature-default.prm.
  TemperatureSolver(const unsigned int order = 2);

  /// Calculate the temperature field

  /// @returns \c true if the final time has been reached
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

  /// Get coordinates of boundary DOFs
  void
  get_boundary_points(const unsigned int       id,
                      std::vector<Point<dim>> &points,
                      std::vector<bool> &      boundary_dofs) const;

  /// Set first-type boundary condition
  void
  set_bc1(const unsigned int id, const double val);

  /// Set thermal radiation and incoming heat flux density boundary condition
  void
  set_bc_rad_mixed(const unsigned int            id,
                   const Vector<double> &        q_in,
                   std::function<double(double)> emissivity,
                   std::function<double(double)> emissivity_deriv);

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

  /// Initialize data before calculation
  void
  prepare_for_solve();

  /// Assemble the system matrix and right-hand-side vector
  void
  assemble_system();


  // Structure that holds scratch data
  struct AssemblyScratchData
  {
    AssemblyScratchData(const Quadrature<dim>     quadrature,
                        const Quadrature<dim - 1> face_quadrature,
                        const FiniteElement<dim> &fe);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);

    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;

    std::vector<double>         lambda_data;
    std::vector<double>         T_q;
    std::vector<double>         T_prev_q;
    std::vector<Tensor<1, dim>> grad_T_q;
    std::vector<double>         T_face_q;
    std::vector<double>         heat_flux_in_face_q;
  };
  // Structure that holds local contributions
  struct AssemblyCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };
  // Local assembly function
  void
  local_assemble_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData &                                 scratch_data,
    AssemblyCopyData &                                    copy_data);
  // Copy local contributions to global
  void
  copy_local_to_global(const AssemblyCopyData &copy_data);


  /// Solve the system of linear equations
  void
  solve_system();

  /// Write temperature at probe points to disk
  void
  output_probes() const;

  Triangulation<dim> triangulation; ///< Mesh
  FE_Q<dim>          fe;            ///< Finite element
  DoFHandler<dim>    dh;            ///< Degrees of freedom

  SparsityPattern      sparsity_pattern; ///< Sparsity pattern
  SparseMatrix<double> system_matrix;    ///< System matrix
  Vector<double>       system_rhs;       ///< Right-hand-side vector

  Vector<double> temperature;        ///< Temperature
  Vector<double> temperature_prev;   ///< Temperature at the previous time step
  Vector<double> temperature_update; ///< Newton update for temperature

  /// Thermal conductivity, W m<sup>-1</sup> K<sup>-1</sup>
  Polynomials::Polynomial<double> lambda;

  /// Data for first-type BC
  std::map<unsigned int, double> bc1_data;
  /// Data for thermal radiation and incoming heat flux density BC
  std::map<unsigned int, heat_flux_data> bc_rad_mixed_data;

  std::vector<Point<dim>> probes; ///< Locations of probe points

  ParameterHandler prm; ///< Parameter handler

  int current_step; ///< Time stepping: index of the current time step
};

template <int dim>
TemperatureSolver<dim>::TemperatureSolver(const unsigned int order)
  : fe(order)
  , dh(triangulation)
  , current_step(0)
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
                    Patterns::Double(),
                    "Max magnitute of Newton update");

  prm.declare_entry("Max Newton iterations",
                    "6",
                    Patterns::Integer(),
                    "Max number of Newton iterations (0 - unlimited)");

  prm.declare_entry("Time step",
                    "1",
                    Patterns::Double(),
                    "Time step in seconds (0 - steady-state)");

  prm.declare_entry("Max time",
                    "10",
                    Patterns::Double(),
                    "Max time in seconds");

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
                    Patterns::Selection(
                      PreconditionSelector<>::get_precondition_names()),
                    "Name of preconditioner");

  prm.declare_entry("Preconditioner relaxation",
                    "1.0",
                    Patterns::Double(0),
                    "Relaxation factor of preconditioner");

  prm.declare_entry("Number of threads",
                    "0",
                    Patterns::Integer(),
                    "Maximum number of threads to be used (0 - default)");

  prm.declare_entry(
    "Output precision",
    "8",
    Patterns::Integer(1),
    "Precision of double variables for output of field and probe data");


  // Physical parameters from https://doi.org/10.1016/S0022-0248(03)01253-3
  prm.declare_entry("Density",
                    "2329",
                    Patterns::Double(0),
                    "Density in kg/m^3");

  prm.declare_entry("Specific heat capacity",
                    "1000",
                    Patterns::Double(0),
                    "Specific heat capacity in J/kg/K");

  prm.declare_entry(
    "Thermal conductivity",
    "98.89, -9.41813870776526E-02, 2.88183040644504E-05",
    Patterns::List(Patterns::Double(), 1),
    "Comma-separated polynomial coefficients (temperature function) in W/m/K^0, W/m/K^1 etc.");

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
  std::cout << "Intializing parameters";

  deallog.depth_console(2);

  // no built-in function exists, parse manually
  std::string              lambda_raw = prm.get("Thermal conductivity");
  std::vector<std::string> lambda_split =
    Utilities::split_string_list(lambda_raw);
  std::vector<double> lambda_coefficients =
    Utilities::string_to_double(lambda_split);

  lambda = Polynomials::Polynomial<double>(lambda_coefficients);

  const auto n_threads = prm.get_integer("Number of threads");
  MultithreadInfo::set_thread_limit(n_threads > 0 ? n_threads :
                                                    MultithreadInfo::n_cores());

  std::cout << "  done\n";

  // some of the parameters are only fetched when needed
  std::cout << "rho=" << prm.get_double("Density") << "\n"
            << "c_p=" << prm.get_double("Specific heat capacity") << "\n";

  const int precision = prm.get_integer("Output precision");
  std::cout << "lambda=" << std::setprecision(precision);
  lambda.print(std::cout);

  std::cout << "n_cores=" << MultithreadInfo::n_cores() << "\n"
            << "n_threads=" << MultithreadInfo::n_threads() << "\n";
}

template <int dim>
bool
TemperatureSolver<dim>::solve()
{
  if (current_step == 0)
    output_probes();

  current_step++;
  const double dt    = get_time_step();
  const double t     = get_time();
  const double t_max = get_max_time();

  temperature_prev = temperature;

  for (int i = 1;; ++i)
    {
      prepare_for_solve();
      assemble_system();
      solve_system();

      // Using step length 1
      temperature += temperature_update;

      // Check convergence
      const double max_abs_dT = temperature_update.linfty_norm();
      std::cout << "Time " << t << " s"
                << "  Newton iteration " << i << "  max T change " << max_abs_dT
                << " K\n";
      if (max_abs_dT < prm.get_double("Max absolute change"))
        break;

      const int N = prm.get_integer("Max Newton iterations");
      if (N >= 1 && i >= N)
        break;
    }

  output_probes();

  if (dt > 0 && t >= t_max)
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
double
TemperatureSolver<dim>::get_time() const
{
  return get_time_step() * current_step;
}

template <int dim>
double
TemperatureSolver<dim>::get_time_step() const
{
  return prm.get_double("Time step");
}

template <int dim>
double
TemperatureSolver<dim>::get_max_time() const
{
  return prm.get_double("Max time");
}

template <int dim>
void
TemperatureSolver<dim>::initialize()
{
  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();

  temperature.reinit(n_dofs);
  temperature_update.reinit(n_dofs);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
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
  const unsigned int n_dofs = dh.n_dofs();
  points.resize(n_dofs);
  boundary_dofs.resize(n_dofs);

  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dh, points);
  DoFTools::extract_boundary_dofs(dh,
                                  ComponentMask(),
                                  boundary_dofs,
                                  {static_cast<types::boundary_id>(id)});
}

template <int dim>
void
TemperatureSolver<dim>::set_bc1(const unsigned int id, const double val)
{
  bc1_data[id] = val;
}

template <int dim>
void
TemperatureSolver<dim>::set_bc_rad_mixed(
  const unsigned int            id,
  const Vector<double> &        q_in,
  std::function<double(double)> emissivity,
  std::function<double(double)> emissivity_deriv)
{
  bc_rad_mixed_data[id].q_in             = q_in;
  bc_rad_mixed_data[id].emissivity       = emissivity;
  bc_rad_mixed_data[id].emissivity_deriv = emissivity_deriv;
}

template <int dim>
void
TemperatureSolver<dim>::add_probe(const Point<dim> &p)
{
  probes.push_back(p);
}

template <int dim>
void
TemperatureSolver<dim>::output_results() const
{
  Timer timer;

  const double t = get_time();

  std::stringstream ss;
  ss << "result-" << dim << "d-order" << fe.degree << "-t" << t << ".vtk";
  const std::string file_name = ss.str();
  std::cout << "Saving to '" << file_name << "'";

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(temperature, "T");
  data_out.add_data_vector(temperature_update, "dT");

  Vector<double> l(temperature.size());
  for (unsigned int i = 0; i < l.size(); ++i)
    {
      l[i] = lambda.value(temperature[i]);
    }
  data_out.add_data_vector(l, "lambda");

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

  data_out.build_patches(fe.degree);

  std::ofstream output(file_name);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  data_out.write_vtk(output);

  std::cout << "  done in " << timer.wall_time() << " s\n";
}

template <int dim>
void
TemperatureSolver<dim>::output_mesh() const
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

  std::cout << "  done in " << timer.wall_time() << " s\n";
}

template <int dim>
void
TemperatureSolver<dim>::output_probes() const
{
  Timer timer;

  std::stringstream ss;
  ss << "probes-" << dim << "d.txt";
  const std::string file_name = ss.str();

  std::cout << "Saving values at probe points to '" << file_name << "'";

  const unsigned int N = probes.size();

  const double t = get_time();

  if (current_step == 0)
    {
      // write header at the first time step
      std::ofstream output(file_name);

      for (unsigned int i = 0; i < N; ++i)
        output << "# probe " << i << ": " << probes[i] << "\n";

      output << "t[s]";
      for (unsigned int i = 0; i < N; ++i)
        output << " T_" << i << "[K]";
      output << "\n";
    }

  Functions::FEFieldFunction<dim> ff(dh, temperature);

  std::vector<double> values(N);
  ff.value_list(probes, values);

  // header is already written, append values at the current time step
  std::ofstream output(file_name, std::ios::app);

  const int precision = prm.get_integer("Output precision");
  output << std::setprecision(precision);

  output << t;
  for (const auto &v : values)
    output << " " << v;
  output << "\n";

  std::cout << "  done in " << timer.wall_time() << " s\n";
}

template <int dim>
void
TemperatureSolver<dim>::prepare_for_solve()
{
  const unsigned int n_dofs = dh.n_dofs();

  temperature_update.reinit(n_dofs);
  system_rhs.reinit(n_dofs);

  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_sparsity_pattern(dh, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  // Apply Dirichlet boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &bc : bc1_data)
    {
      VectorTools::interpolate_boundary_values(dh,
                                               bc.first,
                                               ConstantFunction<dim>(bc.second),
                                               boundary_values);
    }
  for (const auto &bv : boundary_values)
    {
      temperature[bv.first] = bv.second;
    }
}

template <int dim>
TemperatureSolver<dim>::AssemblyScratchData::AssemblyScratchData(
  const Quadrature<dim>     quadrature,
  const Quadrature<dim - 1> face_quadrature,
  const FiniteElement<dim> &fe)
  : fe_values(fe,
              quadrature,
              update_values | update_gradients | update_JxW_values)
  , fe_face_values(fe, face_quadrature, update_values | update_JxW_values)
  , lambda_data(2)
  , T_q(quadrature.size())
  , T_prev_q(quadrature.size())
  , grad_T_q(quadrature.size())
  , T_face_q(face_quadrature.size())
  , heat_flux_in_face_q(face_quadrature.size())
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
  , lambda_data(scratch_data.lambda_data)
  , T_q(scratch_data.T_q)
  , T_prev_q(scratch_data.T_prev_q)
  , grad_T_q(scratch_data.grad_T_q)
  , T_face_q(scratch_data.T_face_q)
  , heat_flux_in_face_q(scratch_data.heat_flux_in_face_q)
{}

template <int dim>
void
TemperatureSolver<dim>::assemble_system()
{
  Timer timer;

  std::cout << "Assembling system";

  const QGauss<dim> &    quadrature(fe.degree + 1 + lambda.degree() / 2);
  const QGauss<dim - 1> &face_quadrature(fe.degree + 1 + lambda.degree() / 2);

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

  std::cout << "  done in " << timer.wall_time() << " s\n";
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
  const double rho    = prm.get_double("Density");
  const double c_p    = prm.get_double("Specific heat capacity");
  const double tmp    = inv_dt * rho * c_p;

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

  std::vector<double> &        lambda_data = scratch_data.lambda_data;
  std::vector<double> &        T_q         = scratch_data.T_q;
  std::vector<double> &        T_prev_q    = scratch_data.T_prev_q;
  std::vector<Tensor<1, dim>> &grad_T_q    = scratch_data.grad_T_q;

  std::vector<types::global_dof_index> &local_dof_indices =
    copy_data.local_dof_indices;

  local_dof_indices.resize(dofs_per_cell);


  fe_values.reinit(cell);

  fe_values.get_function_values(temperature, T_q);
  fe_values.get_function_values(temperature_prev, T_prev_q);
  fe_values.get_function_gradients(temperature, grad_T_q);

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      lambda.value(T_q[q], lambda_data);
      const double lambda_q       = lambda_data[0];
      const double lambda_deriv_q = lambda_data[1];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double &        phi_i      = fe_values.shape_value(i, q);
          const Tensor<1, dim> &grad_phi_i = fe_values.shape_grad(i, q);

          // hand-written optimizations
          const double tmp_i      = tmp * phi_i;
          const double tmp_grad_i = grad_T_q[q] * grad_phi_i;
          const double tmp2_i     = lambda_deriv_q * tmp_grad_i + tmp_i;

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const double &        phi_j      = fe_values.shape_value(j, q);
              const Tensor<1, dim> &grad_phi_j = fe_values.shape_grad(j, q);

              // Newthon's method
              cell_matrix(i, j) +=
                (lambda_q * (grad_phi_j * grad_phi_i) + tmp2_i * phi_j) *
                fe_values.JxW(q);
            }

          cell_rhs(i) -=
            (lambda_q * tmp_grad_i + tmp_i * (T_q[q] - T_prev_q[q])) *
            fe_values.JxW(q);
        }
    }

  for (unsigned int face_number = 0;
       face_number < GeometryInfo<dim>::faces_per_cell;
       ++face_number)
    {
      if (!cell->face(face_number)->at_boundary())
        continue;

      auto it = bc_rad_mixed_data.find(cell->face(face_number)->boundary_id());
      if (it == bc_rad_mixed_data.end())
        continue;

      std::vector<double> &T_face_q = scratch_data.T_face_q;
      std::vector<double> &heat_flux_in_face_q =
        scratch_data.heat_flux_in_face_q;

      fe_face_values.reinit(cell, face_number);

      fe_face_values.get_function_values(temperature, T_face_q);
      fe_face_values.get_function_values(it->second.q_in, heat_flux_in_face_q);

      for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          const double net_heat_flux = sigma_SB *
                                         it->second.emissivity(T_face_q[q]) *
                                         std::pow(T_face_q[q], 4) -
                                       heat_flux_in_face_q[q];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) +=
                    sigma_SB *
                    (it->second.emissivity(T_face_q[q]) * 4.0 *
                       std::pow(T_face_q[q], 3) +
                     it->second.emissivity_deriv(T_face_q[q]) *
                       std::pow(T_face_q[q], 4)) *
                    fe_face_values.shape_value(i, q) *
                    fe_face_values.shape_value(j, q) * fe_face_values.JxW(q);
                }
              cell_rhs(i) -= net_heat_flux * fe_face_values.shape_value(i, q) *
                             fe_face_values.JxW(q);
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
TemperatureSolver<dim>::solve_system()
{
  Timer timer;

  std::cout << "Solving system";

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
      std::cout << "\n";

      const int solver_iterations = prm.get_integer("Linear solver iterations");
      const double solver_tolerance = prm.get_double("Linear solver tolerance");
      IterationNumberControl control(solver_iterations, solver_tolerance);

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
    }

  std::cout << "  done in " << timer.wall_time() << " s\n";
}

#endif