#include <deal.II/base/polynomial.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>

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
  output_results() const;

private:
  Triangulation<dim> triangulation;
  FE_Q<dim>          fe_temp;
  DoFHandler<dim>    dh_temp;

  Vector<double> temperature;

  Polynomials::Polynomial<double> lambda;
};

template <int dim>
TemperatureSolver<dim>::TemperatureSolver(unsigned int order)
  : fe_temp(order)
  , dh_temp(triangulation)
  , lambda(0)
{}

template <int dim>
void
TemperatureSolver<dim>::solve()
{
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
  dh_temp.distribute_dofs(fe_temp);
  const unsigned int n_dofs = dh_temp.n_dofs();
  std::cout << "Number of degrees of freedom: " << n_dofs << "\n";

  AssertDimension(n_dofs, t.size())
  temperature = t;
}

template <int dim>
void
TemperatureSolver<dim>::initialize(double t)
{
  dh_temp.distribute_dofs(fe_temp);
  const unsigned int n_dofs = dh_temp.n_dofs();
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
TemperatureSolver<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dh_temp);
  data_out.add_data_vector(temperature, "T");

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

