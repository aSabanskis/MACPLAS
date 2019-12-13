#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <deal.II/numerics/data_out.h>

using namespace dealii;

class Problem {
public:
	Problem();

	void run();

private:
	void make_grid();
	void initialize();
	void output_results() const;

	Triangulation<2> triangulation;
	FE_Q<2> fe;
	DoFHandler<2> dof_handler;

	Vector<double> temperature;
};

Problem::Problem() :
		fe(2), dof_handler(triangulation) {
}

void Problem::run() {
	make_grid();
	initialize();
	output_results();
}

void Problem::make_grid() {
	const Point<2> center; // all coordinates will be 0

	GridGenerator::hyper_shell(triangulation, center, 0.5, 1.0, 0, true);
	triangulation.refine_global(3);

	std::cout << "Number of active cells: " << triangulation.n_active_cells()
			<< "\n";
}

void Problem::initialize() {
	dof_handler.distribute_dofs(fe);
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< "\n";

	temperature.reinit(dof_handler.n_dofs());
}

void Problem::output_results() const {
	DataOut<2> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(temperature, "T");

	data_out.build_patches();
	std::ofstream output("result.vtk");
	data_out.write_vtk(output);
}

int main() {
	Problem p;
	p.run();

	return 0;
}
