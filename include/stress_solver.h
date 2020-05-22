#ifndef macplas_stress_solver_h
#define macplas_stress_solver_h

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

using namespace dealii;

template <int dim>
class StressSolver
{
public:
  StressSolver(unsigned int order = 2);

  const Triangulation<dim> &
  mesh() const;
  Triangulation<dim> &
  mesh();

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
  Triangulation<dim> triangulation;
  FE_Q<dim>          fe_temp;
  DoFHandler<dim>    dh_temp;

  Vector<double> temperature;

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
const Triangulation<dim> &
StressSolver<dim>::mesh() const
{
  return triangulation;
}

template <int dim>
Triangulation<dim> &
StressSolver<dim>::mesh()
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
  const unsigned int n_dofs = dh_temp.n_dofs();
  std::cout << "Number of temperature degrees of freedom: " << n_dofs << "\n";

  temperature.reinit(n_dofs);
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

  data_out.build_patches();

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

#endif