#ifndef macplas_stress_solver_h
#define macplas_stress_solver_h

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

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

  /// Initialize fields
  void
  initialize();

  /// Get coordinates of DOFs
  void
  get_support_points(std::vector<Point<dim>> &points) const;

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

private:
  /// Initialize physical parameters
  void
  initialize_parameters();

  /// Initialize data before calculation
  void
  prepare_for_solve();

  /// Assemble the system matrix and right-hand-side vector
  void
  assemble_system();

  /// Solve the system of linear equations
  void
  solve_system();

  /// Calculate the stresses from the displacement field
  void
  calculate_stress();

  /// Number of distinct elements of the stress tensor (3D: 6, 2D: 4)
  static const unsigned int n_components = 2 * dim;

  /// Get stiffness tensor
  SymmetricTensor<2, StressSolver<dim>::n_components>
  get_stiffness_tensor() const;

  /// Get strain from FEValues
  Tensor<1, StressSolver<dim>::n_components>
  get_strain(const FEValues<dim> &fe_values,
             const unsigned int & shape_func,
             const unsigned int & q) const;
  /// Get strain from temperature
  Tensor<1, StressSolver<dim>::n_components>
  get_strain(const double &T) const;
  /// Get strain from displacement
  Tensor<1, StressSolver<dim>::n_components>
  get_strain(const std::vector<Tensor<1, dim>> &grad_displacement) const;

  Triangulation<dim> triangulation; ///< Mesh

  FE_Q<dim>       fe_temp;     ///< Finite element for temperature
  DoFHandler<dim> dh_temp;     ///< Degrees of freedom for temperature
  Vector<double>  temperature; ///< Temperature

  FESystem<dim>       fe;              ///< Finite element for displacement
  DoFHandler<dim>     dh;              ///< Degrees of freedom for displacement
  BlockVector<double> displacement;    ///< Displacement
  BlockVector<double> stress;          ///< Stress
  BlockVector<double> stress_deviator; ///< Stress deviator

  Vector<double> stress_hydrostatic; ///< Mean (hydrostatic) stress
  Vector<double> stress_von_Mises;   ///< von Mises stress
  Vector<double> stress_J_2;         ///< Second invariant of deviatoric stress

  BlockSparsityPattern      sparsity_pattern; ///< Sparsity pattern
  BlockSparseMatrix<double> system_matrix;    ///< System matrix
  BlockVector<double>       system_rhs;       ///< Right-hand-side vector

  /// Data for first-type BC
  std::map<unsigned int, std::pair<unsigned int, double>> bc1_data;

  ParameterHandler prm; ///< Parameter handler

  double m_E;     ///< Young's modulus, Pa
  double m_alpha; ///< Thermal expansion coefficient, K<sup>-1</sup>
  double m_nu;    ///< Poisson's ratio, -
  double m_T_ref; ///< Reference temperature, K
  double m_C_11; ///< Second-order elastic constant (stiffness) \f$C_{11}\f$, Pa
  double m_C_12; ///< Second-order elastic constant (stiffness) \f$C_{12}\f$, Pa
  double m_C_44; ///< Second-order elastic constant (stiffness) \f$C_{44}\f$, Pa
};

template <int dim>
StressSolver<dim>::StressSolver(const unsigned int order)
  : fe_temp(order)
  , dh_temp(triangulation)
  , fe(FE_Q<dim>(order), dim)
  , dh(triangulation)
{
  AssertThrow(dim == 3, ExcNotImplemented());

  // Physical parameters from https://doi.org/10.1016/S0022-0248(01)01322-7
  prm.declare_entry("Young's modulus",
                    "1.56e11",
                    Patterns::Double(0),
                    "Young's modulus in Pa");

  prm.declare_entry("Thermal expansion coefficient",
                    "3.2e-6",
                    Patterns::Double(0),
                    "Thermal expansion coefficient in 1/K");

  prm.declare_entry("Poisson's ratio",
                    "0.25",
                    Patterns::Double(0, 0.5),
                    "Poisson's ratio (dimensionless)");

  prm.declare_entry("Reference temperature",
                    "1685",
                    Patterns::Double(),
                    "Reference temperature in K");

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

  m_E     = prm.get_double("Young's modulus");
  m_alpha = prm.get_double("Thermal expansion coefficient");
  m_nu    = prm.get_double("Poisson's ratio");
  m_T_ref = prm.get_double("Reference temperature");

  m_C_11 = m_E * (1 - m_nu) / ((1 + m_nu) * (1 - 2 * m_nu));
  m_C_12 = m_E * m_nu / ((1 + m_nu) * (1 - 2 * m_nu));
  m_C_44 = m_E / (2 * (1 + m_nu));

  std::cout << "  done\n";

  std::cout << "E=" << m_E << "\n"
            << "alpha=" << m_alpha << "\n"
            << "nu=" << m_nu << "\n"
            << "T_ref=" << m_T_ref << "\n"
            << "C_11=" << m_C_11 << "\n"
            << "C_12=" << m_C_12 << "\n"
            << "C_44=" << m_C_44 << "\n";
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

  std::cout << "  done in " << timer() << " s\n";

  std::cout << "dim=" << dim << "\n"
            << "order=" << fe.degree << "\n"
            << "Number of active cells: " << triangulation.n_active_cells()
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

  std::cout << "  done in " << timer() << " s\n";
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

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
void
StressSolver<dim>::prepare_for_solve()
{
  const unsigned int n_dofs_temp = dh_temp.n_dofs();

  system_rhs.reinit(dim, n_dofs_temp);

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
void
StressSolver<dim>::assemble_system()
{
  AssertThrow(dim == 3, ExcNotImplemented());

  Timer timer;

  std::cout << "Assembling system";

  const QGauss<dim> quadrature(fe.degree + 1);

  system_matrix = 0;
  system_rhs    = 0;

  FEValues<dim> fe_values_temp(fe_temp, quadrature, update_values);
  FEValues<dim> fe_values(fe, quadrature, update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<double> T_q(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const SymmetricTensor<2, n_components> stiffness = get_stiffness_tensor();

  typename DoFHandler<dim>::active_cell_iterator cell_temp =
                                                   dh_temp.begin_active(),
                                                 endc_temp = dh_temp.end(),
                                                 cell      = dh.begin_active(),
                                                 endc      = dh.end();
  for (; cell != endc; ++cell_temp, ++cell)
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values_temp.reinit(cell_temp);
      fe_values.reinit(cell);

      fe_values_temp.get_function_values(temperature, T_q);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<1, n_components> epsilon_T = get_strain(T_q[q]);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, n_components> strain_i =
                get_strain(fe_values, i, q);

              const Tensor<1, n_components> strain_i_stiffness =
                strain_i * stiffness;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, n_components> strain_j =
                    get_strain(fe_values, j, q);

                  cell_matrix(i, j) +=
                    (strain_i_stiffness * strain_j) * fe_values.JxW(q);
                }
              cell_rhs(i) +=
                (strain_i_stiffness * epsilon_T) * fe_values.JxW(q);
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

  // Apply boundary conditions for displacement
  std::map<types::global_dof_index, double> boundary_values;
  for (auto const &it : bc1_data)
    {
      std::vector<bool> mask(dim, false);
      mask[it.second.first] = true;

      VectorTools::interpolate_boundary_values( // break line
        dh,
        it.first,
        ConstantFunction<dim>(it.second.second),
        boundary_values,
        mask);
    }

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     displacement,
                                     system_rhs);

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
void
StressSolver<dim>::solve_system()
{
  Timer timer;

  std::cout << "Solving system";

  SparseDirectUMFPACK A;
  A.initialize(system_matrix);
  A.vmult(displacement, system_rhs);

  std::cout << "  done in " << timer() << " s\n";
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

  std::vector<std::vector<Tensor<1, dim>>> grad_displacement_q(
    n_q_points, std::vector<Tensor<1, dim>>(dim));

  std::vector<Vector<double>> stress_q(n_components,
                                       Vector<double>(n_q_points));

  std::vector<Vector<double>> stress_cell(n_components,
                                          Vector<double>(dofs_per_cell_temp));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_temp);

  const SymmetricTensor<2, n_components> stiffness = get_stiffness_tensor();

  typename DoFHandler<dim>::active_cell_iterator cell_temp =
                                                   dh_temp.begin_active(),
                                                 endc_temp = dh_temp.end(),
                                                 cell      = dh.begin_active(),
                                                 endc      = dh.end();
  for (; cell != endc; ++cell_temp, ++cell)
    {
      fe_values_temp.reinit(cell_temp);
      fe_values.reinit(cell);

      fe_values_temp.get_function_values(temperature, T_q);

      fe_values.get_function_gradients(displacement, grad_displacement_q);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<1, n_components> epsilon_T = get_strain(T_q[q]);
          const Tensor<1, n_components> epsilon_e =
            get_strain(grad_displacement_q[q]);

          const Tensor<1, n_components> s = stiffness * (epsilon_e - epsilon_T);

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

  std::cout << "  done in " << timer() << " s\n";
}

template <int dim>
SymmetricTensor<2, StressSolver<dim>::n_components>
StressSolver<dim>::get_stiffness_tensor() const
{
  AssertThrow(dim == 3, ExcNotImplemented());

  SymmetricTensor<2, n_components> tmp;
  tmp[0][0] = tmp[1][1] = tmp[2][2] = m_C_11;
  tmp[3][3] = tmp[4][4] = tmp[5][5] = m_C_44;
  tmp[2][1] = tmp[2][0] = tmp[1][0] = m_C_12;

  return tmp;
}

template <int dim>
Tensor<1, StressSolver<dim>::n_components>
StressSolver<dim>::get_strain(const FEValues<dim> &fe_values,
                              const unsigned int & shape_func,
                              const unsigned int & q) const
{
  AssertThrow(dim == 3, ExcNotImplemented());

  Tensor<1, n_components> tmp;

  tmp[0] = fe_values.shape_grad_component(shape_func, q, 0)[0];
  tmp[1] = fe_values.shape_grad_component(shape_func, q, 1)[1];
  tmp[2] = fe_values.shape_grad_component(shape_func, q, 2)[2];

  tmp[3] = fe_values.shape_grad_component(shape_func, q, 2)[1] +
           fe_values.shape_grad_component(shape_func, q, 1)[2];

  tmp[4] = fe_values.shape_grad_component(shape_func, q, 2)[0] +
           fe_values.shape_grad_component(shape_func, q, 0)[2];

  tmp[5] = fe_values.shape_grad_component(shape_func, q, 1)[0] +
           fe_values.shape_grad_component(shape_func, q, 0)[1];

  return tmp;
}

template <int dim>
Tensor<1, StressSolver<dim>::n_components>
StressSolver<dim>::get_strain(const double &T) const
{
  AssertThrow(dim == 3, ExcNotImplemented());

  Tensor<1, n_components> tmp;
  tmp[0] = tmp[1] = tmp[2] = m_alpha * (T - m_T_ref);
  tmp[3] = tmp[4] = tmp[5] = 0;

  return tmp;
}

template <int dim>
Tensor<1, StressSolver<dim>::n_components>
StressSolver<dim>::get_strain(
  const std::vector<Tensor<1, dim>> &grad_displacement) const
{
  AssertThrow(dim == 3, ExcNotImplemented());
  Tensor<1, n_components> tmp;

  tmp[0] = grad_displacement[0][0];
  tmp[1] = grad_displacement[1][1];
  tmp[2] = grad_displacement[2][2];

  tmp[3] = grad_displacement[2][1] + grad_displacement[1][2];
  tmp[4] = grad_displacement[2][0] + grad_displacement[0][2];
  tmp[5] = grad_displacement[1][0] + grad_displacement[0][1];

  return tmp;
}

#endif