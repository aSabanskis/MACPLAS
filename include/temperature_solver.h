#ifndef macplas_temperature_solver_h
#define macplas_temperature_solver_h

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
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
  TemperatureSolver(unsigned int order = 2);

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

  /// Initialize temperature field
  void
  initialize(const Vector<double> &T);

  /// Initialize temperature field
  void
  initialize(double T);

  /// Initialize temperature-dependent thermal conductivity
  void
  initialize(const Polynomials::Polynomial<double> &l);

  /// Get coordinates of boundary DOFs
  void
  get_boundary_points(unsigned int             id,
                      std::vector<Point<dim>> &points,
                      std::vector<bool> &      boundary_dofs) const;

  /// Set first-type boundary condition
  void
  set_bc1(unsigned int id, double val);

  /// Set thermal radiation and incoming heat flux density boundary condition
  void
  set_bc_rad_mixed(unsigned int                  id,
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
  /// Initialize data before calculation
  void
  prepare_for_solve();

  /// Assemble the system matrix and right-hand-side vector
  void
  assemble_system();

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

  bool first;        ///< Time stepping: flag for the first time step
  int  current_step; ///< Time stepping: index of the current time step
};

template <int dim>
TemperatureSolver<dim>::TemperatureSolver(unsigned int order)
  : fe(order)
  , dh(triangulation)
  , lambda(0)
  , first(true)
  , current_step(0)
{
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

  prm.declare_entry(
    "Output precision",
    "8",
    Patterns::Integer(1),
    "Precision of double variables for output of field and probe data");

  prm.declare_entry("Density", "1000", Patterns::Double(), "Density in kg/m^3");

  prm.declare_entry("Specific heat capacity",
                    "1000",
                    Patterns::Double(),
                    "Specific heat capacity in J/kg/K");

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
}

template <int dim>
bool
TemperatureSolver<dim>::solve()
{
  current_step++;
  const double dt = prm.get_double("Time step");
  const double t  = dt * current_step;

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

  if (dt > 0 && t >= prm.get_double("Max time"))
    return false;

  first = false;

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
void
TemperatureSolver<dim>::initialize(const Vector<double> &T)
{
  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();

  AssertDimension(n_dofs, T.size());
  temperature = T;

  std::cout << "dim=" << dim << "\n"
            << "order=" << fe.degree << "\n"
            << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
            << "Number of degrees of freedom for temperature: " << n_dofs
            << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::initialize(double T)
{
  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();

  temperature.reinit(n_dofs);
  temperature_update.reinit(n_dofs);
  temperature.add(T);

  std::cout << "dim=" << dim << "\n"
            << "order=" << fe.degree << "\n"
            << "Number of active cells: " << triangulation.n_active_cells()
            << "\n"
            << "Number of degrees of freedom for temperature: " << n_dofs
            << "\n";
}

template <int dim>
void
TemperatureSolver<dim>::initialize(const Polynomials::Polynomial<double> &l)
{
  lambda = l;
}

template <int dim>
void
TemperatureSolver<dim>::get_boundary_points(
  unsigned int             id,
  std::vector<Point<dim>> &points,
  std::vector<bool> &      boundary_dofs) const
{
  const unsigned int n_dofs = dh.n_dofs();
  points.resize(n_dofs);
  boundary_dofs.resize(n_dofs);

  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dh, points);
  DoFTools::extract_boundary_dofs(dh, ComponentMask(), boundary_dofs, {id});
}

template <int dim>
void
TemperatureSolver<dim>::set_bc1(unsigned int id, double val)
{
  bc1_data[id] = val;
}

template <int dim>
void
TemperatureSolver<dim>::set_bc_rad_mixed(
  unsigned int                  id,
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

  const double dt = prm.get_double("Time step");
  const double t  = dt * current_step;

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
                                      {data.first});

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

  std::cout << "  done in " << timer() << " s\n";
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

  std::cout << "  done in " << timer() << " s\n";
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

  const double dt = prm.get_double("Time step");
  const double t  = dt * current_step;

  if (first)
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

  std::cout << "  done in " << timer() << " s\n";
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
void
TemperatureSolver<dim>::assemble_system()
{
  Timer timer;

  std::cout << "Assembling system";

  const double dt     = prm.get_double("Time step");
  const double inv_dt = dt == 0 ? 0 : 1 / dt;

  const double rho = prm.get_double("Density");
  const double c_p = prm.get_double("Specific heat capacity");

  const QGauss<dim>     quadrature(fe.degree + 1 + lambda.degree() / 2);
  const QGauss<dim - 1> face_quadrature(fe.degree + 1 + lambda.degree() / 2);

  system_matrix = 0;
  system_rhs    = 0;

  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe,
                                   face_quadrature,
                                   update_values | update_JxW_values);

  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int n_q_points      = quadrature.size();
  const unsigned int n_face_q_points = face_quadrature.size();


  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<double>         lambda_data(2);
  std::vector<double>         T_q(n_q_points);
  std::vector<double>         T_prev_q(n_q_points);
  std::vector<Tensor<1, dim>> grad_T_q(n_q_points);
  std::vector<double>         T_face_q(n_face_q_points);
  std::vector<double>         heat_flux_in_face_q(n_face_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = dh.begin_active(),
                                                 endc = dh.end();
  for (; cell != endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      fe_values.get_function_values(temperature, T_q);
      fe_values.get_function_values(temperature_prev, T_prev_q);
      fe_values.get_function_gradients(temperature, grad_T_q);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          lambda.value(T_q[q], lambda_data);
          const double lambda_q      = lambda_data[0];
          const double grad_lambda_q = lambda_data[1];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double &        phi_i      = fe_values.shape_value(i, q);
              const Tensor<1, dim> &grad_phi_i = fe_values.shape_grad(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const double &        phi_j = fe_values.shape_value(j, q);
                  const Tensor<1, dim> &grad_phi_j = fe_values.shape_grad(j, q);

                  // Newthon's method
                  cell_matrix(i, j) += ((lambda_q * grad_phi_j +
                                         grad_lambda_q * phi_j * grad_T_q[q]) *
                                          grad_phi_i +
                                        inv_dt * rho * c_p * phi_j * phi_i) *
                                       fe_values.JxW(q);
                }

              cell_rhs(i) -=
                (lambda_q * grad_T_q[q] * grad_phi_i +
                 inv_dt * rho * c_p * (T_q[q] - T_prev_q[q]) * phi_i) *
                fe_values.JxW(q);
            }
        }

      for (unsigned int face_number = 0;
           face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          if (!cell->face(face_number)->at_boundary())
            continue;

          auto it =
            bc_rad_mixed_data.find(cell->face(face_number)->boundary_id());
          if (it == bc_rad_mixed_data.end())
            continue;

          fe_face_values.reinit(cell, face_number);

          fe_face_values.get_function_values(temperature, T_face_q);
          fe_face_values.get_function_values(it->second.q_in,
                                             heat_flux_in_face_q);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              const double net_heat_flux =
                sigma_SB * it->second.emissivity(T_face_q[q]) *
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
                        fe_face_values.shape_value(j, q) *
                        fe_face_values.JxW(q);
                    }
                  cell_rhs(i) -= net_heat_flux *
                                 fe_face_values.shape_value(i, q) *
                                 fe_face_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

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

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
void
TemperatureSolver<dim>::solve_system()
{
  Timer timer;

  std::cout << "Solving system";

  SparseDirectUMFPACK A;
  A.initialize(system_matrix);
  A.vmult(temperature_update, system_rhs);

  std::cout << "  done in " << timer() << " s\n";
}

#endif