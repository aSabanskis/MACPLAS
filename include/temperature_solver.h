#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/polynomial.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

using namespace dealii;

template <int dim>
class TemperatureSolver
{
public:
  TemperatureSolver(unsigned int order = 2);

  void
  solve();

  const Triangulation<dim> &mesh() const;
  Triangulation<dim> &mesh();

  void
  initialize(const Vector<double> &t);

  void
  initialize(double t);

  void
  initialize(const Polynomials::Polynomial<double> &l);

  void
  set_bc1(unsigned int id, double val);

  void
  output_results() const;

private:
  void prepare_for_solve();

  void assemble_system();

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dh;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       system_rhs;

  Vector<double> temperature;
  Vector<double> temperature_update;

  Polynomials::Polynomial<double> lambda;

  // boundary condition data
  std::map<unsigned int, double> bc1_data;
};

template <int dim>
TemperatureSolver<dim>::TemperatureSolver(unsigned int order)
  : fe(order)
  , dh(triangulation)
  , lambda(0)
{}

template <int dim>
void
TemperatureSolver<dim>::solve()
{
  for (int i=0; i<6; ++i)
  {
    prepare_for_solve();
    assemble_system();

    SparseDirectUMFPACK A;
    A.initialize(system_matrix);
    A.vmult(temperature_update, system_rhs);

    // Using step length 1
    temperature += temperature_update;
    std::cout << "Newton iteration " << i
              << " max change: " << temperature_update.linfty_norm()
              << "\n";
  }
}

template <int dim>
const Triangulation<dim> &
TemperatureSolver<dim>::mesh() const
{
  return triangulation;
}

template <int dim>
Triangulation<dim> &
TemperatureSolver<dim>::mesh()
{
  return triangulation;
}

template <int dim>
void
TemperatureSolver<dim>::initialize(const Vector<double> &t)
{
  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();
  std::cout << "Number of degrees of freedom: " << n_dofs << "\n";

  AssertDimension(n_dofs, t.size())
  temperature = t;
}

template <int dim>
void
TemperatureSolver<dim>::initialize(double t)
{
  dh.distribute_dofs(fe);
  const unsigned int n_dofs = dh.n_dofs();
  std::cout << "Number of degrees of freedom: " << n_dofs << "\n";

  temperature.reinit(n_dofs);
  temperature.add(t);
}

template <int dim>
void
TemperatureSolver<dim>::initialize(const Polynomials::Polynomial<double> &l)
{
  lambda = l;
}

template <int dim>
void
TemperatureSolver<dim>::set_bc1(unsigned int id, double val)
{
  bc1_data[id] = val;
}

template <int dim>
void
TemperatureSolver<dim>::output_results() const
{
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

  data_out.build_patches();

  const std::string file_name = "result-" + std::to_string(dim) + "d.vtk";
  std::cout << "Saving to " << file_name << "\n";

  std::ofstream output(file_name);
  data_out.write_vtk(output);
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

  // Apply dirichlet boundary conditions
  std::map<types::global_dof_index,double> boundary_values;
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
  const QGauss<dim> quadrature(fe.degree+1);

  system_matrix = 0;
  system_rhs = 0;

  FEValues<dim> fe_values(fe, quadrature,
                          update_values            |
                          update_gradients         |
                          update_quadrature_points |
                          update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<double>        lambda_data(2);
  std::vector<double>        T_q(n_q_points);
  std::vector<Tensor<1,dim>> grad_T_q(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dh.begin_active(),
  endc = dh.end();
  for (; cell!=endc; ++cell)
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    fe_values.get_function_values(temperature, T_q);
    fe_values.get_function_gradients(temperature, grad_T_q);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      lambda.value(temperature[q], lambda_data);
      const double lambda_q = lambda_data[0];
      const double grad_lambda_q = lambda_data[1];

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        const Tensor<1, dim> &grad_phi_i = fe_values.shape_grad(i, q);

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const double &phi_j = fe_values.shape_value(j, q);
          const Tensor<1, dim> &grad_phi_j = fe_values.shape_grad(j, q);

          // Newthon's method
          cell_matrix(i, j) += (
                                 (
                                  lambda_q * grad_phi_j
                                  +
                                  grad_lambda_q * phi_j* grad_T_q[q]
                                  )
                                  * grad_phi_i
                                )
                                * fe_values.JxW(q);
        }

        cell_rhs(i) -= (
                        lambda_q * grad_T_q[q] * grad_phi_i
                        )
                        * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i],
                          local_dof_indices[j],
                          cell_matrix(i,j));

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }

  // Apply boundary conditions for Newton update
  std::map<types::global_dof_index,double> boundary_values;
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
}
