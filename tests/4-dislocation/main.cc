#include "../../include/dislocation_solver.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem();

  void
  run();

private:
  DislocationSolver<dim> solver;
};

template <int dim>
Problem<dim>::Problem()
{}

template <int dim>
void
Problem<dim>::run()
{
  // do not calculate
}

int
main()
{
  Problem<3> p3d;
  p3d.run();

  return 0;
}
