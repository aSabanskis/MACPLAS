#ifndef macplas_stress_solver_h
#define macplas_stress_solver_h

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

template <int dim>
class StressSolver
{
public:
  StressSolver(unsigned int order = 2);

  void
  solve();

  const Triangulation<dim> &
  get_mesh() const;
  Triangulation<dim> &
  get_mesh();

  const Vector<double> &
  get_temperature() const;
  Vector<double> &
  get_temperature();

  void
  initialize();

  void
  get_support_points(std::vector<Point<dim>> &points) const;

  void
  output_results() const;

  void
  output_mesh() const;

private:
  void
  prepare_for_solve();

  void
  assemble_system();

  void
  solve_system();

  Triangulation<dim> triangulation;

  FE_Q<dim>       fe_temp;
  DoFHandler<dim> dh_temp;
  Vector<double>  temperature;

  FESystem<dim>       fe;
  DoFHandler<dim>     dh;
  BlockVector<double> displacement;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  BlockVector<double>       system_rhs;

  // Parameters
  ParameterHandler prm;

  // Young's modulus, Pa
  double m_E;
  // Thermal expansion coefficient, 1/K
  double m_alpha;
  // Poisson's ratio, -
  double m_nu;
};

template <int dim>
StressSolver<dim>::StressSolver(unsigned int order)
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

  m_E     = prm.get_double("Young's modulus");
  m_alpha = prm.get_double("Thermal expansion coefficient");
  m_nu    = prm.get_double("Poisson's ratio");
}

template <int dim>
void
StressSolver<dim>::solve()
{
  prepare_for_solve();
  assemble_system();
  solve_system();
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
  dh_temp.distribute_dofs(fe_temp);
  dh.distribute_dofs(fe);

  const unsigned int n_dofs_temp = dh_temp.n_dofs();
  std::cout << "Number of degrees of freedom for temperature: " << n_dofs_temp
            << "\n";

  temperature.reinit(n_dofs_temp);
  displacement.reinit(dim, n_dofs_temp);
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
StressSolver<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh_temp);
  data_out.add_data_vector(temperature, "T");

  for (unsigned int i = 0; i < displacement.n_blocks(); ++i)
    {
      const std::string name = "displacement_" + std::to_string(i);
      data_out.add_data_vector(displacement.block(i), name);
    }

  data_out.build_patches(fe.degree);

  const std::string file_name = "result-" + std::to_string(dim) + "d.vtk";
  std::cout << "Saving to " << file_name << "\n";

  std::ofstream output(file_name);
  data_out.write_vtk(output);
}

template <int dim>
void
StressSolver<dim>::output_mesh() const
{
  std::stringstream ss;
  ss << "mesh-" << dim << "d.msh";
  const std::string file_name = ss.str();
  std::cout << "Saving to " << file_name << "\n";

  std::ofstream output(file_name);

  GridOut grid_out;
  grid_out.set_flags(GridOutFlags::Msh(true));
  grid_out.write_msh(triangulation, output);
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

  DoFTools::make_sparsity_pattern(dh, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void
StressSolver<dim>::assemble_system()
{
  system_matrix = 0;
  system_rhs    = 0;
}

template <int dim>
void
StressSolver<dim>::solve_system()
{}

#endif