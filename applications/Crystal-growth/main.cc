#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <algorithm>
#include <fstream>

#include "../../include/dislocation_solver.h"
#include "../../include/temperature_solver.h"
#include "../../include/utilities.h"

using namespace dealii;

template <int dim>
class Problem
{
public:
  Problem(const unsigned int order = 2, const bool use_default_prm = false);

  void
  run();

private:
  void
  make_grid();

  void
  deform_grid();

  void
  calculate_field_gradients();

  void
  calculate_dof_distance();

  void
  update_fields();

  void
  initialize_temperature();

  void
  initialize_dislocation();

  void
  set_temperature_BC();

  /** Reduces the current time step when the DOF movement is too large
   */
  void
  update_grid_time_step();

  /** Checks whether the results need to be outputted at the end of the current
   * time step, and adjusts it if necessary */
  bool
  update_output_time_step();

  void
  set_time_step(const double dt);

  void
  solve_steady_temperature();

  void
  solve_temperature();

  void
  solve_temperature_dislocation();

  void
  output_results(const bool data     = false,
                 const bool vtk      = true,
                 const bool boundary = dim == 2,
                 const bool mesh     = true) const;

  // false if only the temperature field is calculated
  bool
  with_dislocation() const;

  Timer timer;

  TemperatureSolver<dim> temperature_solver;

  DislocationSolver<dim> dislocation_solver;

  std::vector<Point<dim>> support_points, support_points_prev;

  // DOF shift
  std::vector<Tensor<1, dim>> shift;

  // DOF shift relative to the distance to neighbour
  Vector<double> shift_relative;

  // distance between closest DOFs (for time step control)
  Vector<double> dof_distance;

  // for testing of interpolation
  Vector<double> f_test;

  // crystallization interface
  constexpr static unsigned int boundary_id_interface = 0;
  // crystal side surface
  constexpr static unsigned int boundary_id_surface = 1;
  // crystal axis
  constexpr static unsigned int boundary_id_axis = 2;

  unsigned int point_id_axis_z_min, point_id_axis_z_max, point_id_triple;

  SurfaceInterpolator2D surface_projector;

  std::unique_ptr<Function<1>> ambient_temperature;

  std::unique_ptr<Function<1>> pull_rate;

  std::unique_ptr<Function<1>> crystal_radius;

  FunctionParser<2> interface_shape;

  DoFGradientEvaluation<dim> grad_eval;

  double previous_time_step;
  double next_output_time;

  ParameterHandler prm;
};

template <int dim>
Problem<dim>::Problem(const unsigned int order, const bool use_default_prm)
  : temperature_solver(order, use_default_prm)
  , dislocation_solver(order, use_default_prm)
{
  prm.declare_entry("Melting point",
                    "1210",
                    Patterns::Double(0),
                    "Melting point T_0 in K");

  prm.declare_entry("Max temperature change",
                    "0.1",
                    Patterns::Double(0),
                    "Maximum temperature change in K");

  prm.declare_entry("Emissivity",
                    "0.55",
                    Patterns::Double(0),
                    "Emissivity epsilon (dimensionless)");

  prm.declare_entry(
    "Ambient temperature",
    "300",
    Patterns::Anything(),
    "Ambient temperature T_amb in K (vertical coordinate z function or data file name)");

  prm.declare_entry("Temperature only",
                    "false",
                    Patterns::Bool(),
                    "Calculate just the temperature field");

  prm.declare_entry(
    "Pull rate",
    "1e-5",
    Patterns::Anything(),
    "Crystal pull rate V in m/s (time function or data file name)");

  prm.declare_entry(
    "Crystal radius",
    "0.01",
    Patterns::Anything(),
    "Crystal radius R in m (length function or data file name)");

  prm.declare_entry(
    "Interface shape",
    "0",
    Patterns::Anything(),
    "Crystallization interface shape z(r, t) in laboratory reference frame");

  prm.declare_entry(
    "Surface update tolerance",
    "0.1e-3",
    Patterns::Double(0),
    "Distance in m for update of precise crystal shape (0 - always)");

  prm.declare_entry(
    "Interpolation test function",
    "",
    Patterns::Anything(),
    "Function f(r, z) for interpolation test on a moving mesh (empty - disabled)");

  prm.declare_entry(
    "Max relative shift",
    "0",
    Patterns::Double(0),
    "Maximum relative (to neighbour) mesh movement (0 - unlimited)");

  prm.declare_entry("Laplace tolerance",
                    "1e-10",
                    Patterns::Double(0),
                    "Tolerance of linear solver for Laplace transform");

  prm.declare_entry("Max time",
                    "0",
                    Patterns::Double(0),
                    "Maximum time in seconds (0 - use solver parameters)");

  prm.declare_entry("Output time step",
                    "0",
                    Patterns::Double(0),
                    "Time interval in s between result output (0 - disabled). "
                    "The time step is adjusted to match the required time.");

  prm.declare_entry("Probe coordinates x",
                    "0, 0.01",
                    Patterns::List(Patterns::Double(), 1),
                    "Comma-separated radial coordinates");

  prm.declare_entry("Probe coordinates z",
                    "0.002, 0.002",
                    Patterns::List(Patterns::Double(), 1),
                    "Comma-separated vertical coordinates");

  if (use_default_prm)
    {
      std::ofstream of("problem.prm");
      prm.print_parameters(of, ParameterHandler::Text);
    }
  else
    try
      {
        prm.parse_input("problem.prm");
      }
    catch (std::exception &e)
      {
        std::cout << e.what() << "\n";

        std::ofstream of("problem-default.prm");
        prm.print_parameters(of, ParameterHandler::Text);
      }

  const double t_max = prm.get_double("Max time");
  if (t_max > 0)
    {
      temperature_solver.get_parameters().set("Max time", t_max);
      dislocation_solver.get_parameters().set("Max time", t_max);
    }

  // print all problem-specific and solver parameters
  std::cout << "# ---------------------\n"
            << "# Problem\n";
  prm.print_parameters(std::cout, ParameterHandler::Text);

  std::cout << "# ---------------------\n"
            << "# " << temperature_solver.solver_name() << '\n';
  temperature_solver.get_parameters().print_parameters(std::cout,
                                                       ParameterHandler::Text);

  if (with_dislocation())
    {
      std::cout << "# ---------------------\n"
                << "# " << dislocation_solver.solver_name() << '\n';
      dislocation_solver.get_parameters().print_parameters(
        std::cout, ParameterHandler::Text);

      std::cout << "# ---------------------\n"
                << "# " << dislocation_solver.get_stress_solver().solver_name()
                << '\n';
      dislocation_solver.get_stress_solver().get_parameters().print_parameters(
        std::cout, ParameterHandler::Text);
    }

  initialize_function(ambient_temperature, prm.get("Ambient temperature"), "z");

  initialize_function(pull_rate, prm.get("Pull rate"));

  initialize_function(crystal_radius, prm.get("Crystal radius"), "L");

  interface_shape.initialize("r,t",
                             prm.get("Interface shape"),
                             typename FunctionParser<2>::ConstMap());

  next_output_time = prm.get_double("Output time step");
}

template <int dim>
void
Problem<dim>::run()
{
  // initialize the mesh, all fields (steady T) and output the results at t=0

  make_grid();

  initialize_temperature();

  calculate_dof_distance();

  solve_steady_temperature();

  if (with_dislocation())
    initialize_dislocation();

  calculate_field_gradients();

  output_results();

  // now run transient simulations unless the time step is zero
  if (temperature_solver.get_time_step() == 0)
    {
      std::cout << "dt=0, stopping simulation\n";
    }
  else if (!with_dislocation())
    {
      solve_temperature();
    }
  else
    {
      solve_temperature_dislocation();
    }

  std::cout << "Finished in " << timer.wall_time() << " s\n";
}

template <int dim>
void
Problem<dim>::make_grid()
{
  Triangulation<dim> &triangulation = temperature_solver.get_mesh();

  GridIn<dim> gi;
  gi.attach_triangulation(triangulation);
  std::ifstream f("mesh-" + std::to_string(dim) + "d.msh");
  gi.read_msh(f);

  if (with_dislocation())
    dislocation_solver.get_mesh().copy_triangulation(triangulation);

  // print info
  std::cout << "Number of cells: " << triangulation.n_cells() << "\n"
            << "Number of points: " << triangulation.n_vertices() << "\n";

  const auto bounds = triangulation.get_boundary_ids();
  for (const auto &b : bounds)
    {
      const auto points = get_boundary_points(triangulation, b);
      std::cout << "Boundary No. " << (int)b << " : " << points.size()
                << " points\n";
    }

  // preprocess some point indices
  auto cmp_z = [](const std::pair<unsigned int, Point<dim>> &it1,
                  const std::pair<unsigned int, Point<dim>> &it2) {
    return it1.second[dim - 1] < it2.second[dim - 1];
  };

  const auto points_axis = get_boundary_points(triangulation, boundary_id_axis),
             points_surface =
               get_boundary_points(triangulation, boundary_id_surface);
  AssertThrow(!points_axis.empty(),
              ExcMessage("No points on boundary No. " +
                         std::to_string(boundary_id_axis) + " (axis) found"));
  AssertThrow(!points_surface.empty(),
              ExcMessage("No points on boundary No. " +
                         std::to_string(boundary_id_surface) +
                         " (side surface) found"));

  point_id_axis_z_min =
    std::min_element(points_axis.begin(), points_axis.end(), cmp_z)->first;
  point_id_axis_z_max =
    std::max_element(points_axis.begin(), points_axis.end(), cmp_z)->first;
  point_id_triple =
    std::min_element(points_surface.begin(), points_surface.end(), cmp_z)
      ->first;

  std::cout << "Axis lowest point = " << point_id_axis_z_min << '\n'
            << "Axis highest point = " << point_id_axis_z_max << '\n'
            << "Triple point = " << point_id_triple << '\n';

  // save the crystal side surface
  std::vector<Point<dim>> points_sorted;
  for (const auto &it : points_surface)
    {
      points_sorted.push_back(it.second);
    }
  std::sort(points_sorted.begin(),
            points_sorted.end(),
            [](const Point<dim> &p1, const Point<dim> &p2) {
              // from highest to lowest, so that additional points can be added
              return p1[dim - 1] > p2[dim - 1];
            });
#ifdef DEBUG
  std::cout << "points_sorted.front = " << points_sorted.front()
            << " points_sorted.back = " << points_sorted.back() << '\n';
#endif
  surface_projector.set_points(points_sorted);

  const double L  = (points_surface.at(point_id_axis_z_max) -
                    points_surface.at(point_id_triple))[dim - 1];
  const double R0 = crystal_radius->value(Point<1>(L));
  const double R  = points_surface.at(point_id_triple)[0];

  AssertThrow(std::fabs(R - R0) < 1e-4,
              ExcMessage("The actual radius R = " + std::to_string(R) +
                         " m differs from the expected R0 = " +
                         std::to_string(R0) + " m"));

  temperature_solver.add_output("L[m]", L);
  temperature_solver.add_output("R[m]", R);
  if (with_dislocation())
    {
      dislocation_solver.add_output("L[m]", L);
      dislocation_solver.add_output("R[m]", R);
    }
}

template <int dim>
void
Problem<dim>::deform_grid()
{
  Timer timer;

  std::cout << "Deforming grid";

  // the mesh will be modified, make a copy
  Triangulation<dim> triangulation;
  triangulation.copy_triangulation(temperature_solver.get_mesh());

  temperature_solver.get_support_points(support_points_prev);
  calculate_field_gradients();

  calculate_dof_distance();

  const auto &points = triangulation.get_vertices();

  const auto points_axis = get_boundary_points(triangulation, boundary_id_axis),
             points_surface =
               get_boundary_points(triangulation, boundary_id_surface),
             points_interface =
               get_boundary_points(triangulation, boundary_id_interface);

  const Point<dim> p_axis_1 = points.at(point_id_axis_z_min),
                   p_axis_2 = points.at(point_id_axis_z_max),
                   p_triple = points.at(point_id_triple);

  const double t  = temperature_solver.get_time();
  const double dt = temperature_solver.get_time_step();
  const double V  = pull_rate->value(Point<1>(t + dt));

  // Vertical shift of the crystal due to the pull rate
  Point<dim> dz;
  dz[dim - 1] = V * dt;

  const double L  = (p_axis_2 - p_triple)[dim - 1];
  const double dL = V * dt + p_triple[dim - 1] -
                    interface_shape.value(Point<2>(p_triple[0], t + dt));
  const double R  = p_triple[0];
  const double R0 = crystal_radius->value(Point<1>(L + dL)); // at the end

  temperature_solver.add_output("L[m]", L + dL);
  temperature_solver.add_output("R[m]", R0);
  temperature_solver.add_output("V[m/s]", V);
  if (with_dislocation())
    {
      dislocation_solver.add_output("L[m]", L + dL);
      dislocation_solver.add_output("R[m]", R0);
      dislocation_solver.add_output("V[m/s]", V);
    }

  auto calc_interface_displacement = [&](const Point<dim> &p) {
    Point<dim> dp;

    // Scale radially
    dp[0] = (R0 - R) * (p[0] / R);

    // Shift vertically. The crystal pulling has to taken into account to keep
    // the correct interface position at the end of the time step.
    dp[dim - 1] =
      interface_shape.value(Point<2>(p[0], t + dt)) - dz[dim - 1] - p[dim - 1];

    return dp;
  };

  const auto dp_axis   = calc_interface_displacement(p_axis_1);
  const auto dp_triple = calc_interface_displacement(p_triple);

#ifdef DEBUG
  std::cout << '\n'
            << "p_axis = " << p_axis_1 << " dp_axis = " << dp_axis << '\n'
            << "p_triple = " << p_triple << " dp_triple = " << dp_triple
            << '\n';
#endif

  std::vector<Point<dim>> projector_points = surface_projector.get_points();
  const Point<dim>        p_triple_old     = projector_points.back();
  const Point<dim>        p_triple_new     = p_triple + dp_triple;

  auto points_new = points_axis;
  for (auto &it : points_new)
    {
      const auto dp =
        dp_axis * (p_axis_2 - it.second).norm() / (p_axis_2 - p_axis_1).norm();

      it.second += dp;
    }

  // Special treatment for the crystal side surface: project to the precise
  // surface shape, which needs to be updated with the new triple point position
  projector_points.push_back(p_triple_new);
  surface_projector.set_points(projector_points);

#ifdef DEBUG
  std::stringstream tmp;
  tmp << std::setprecision(8) << "crystal-surface-t" << t << ".dat";
  std::ofstream out(tmp.str());
  // output the precise crystal shape
  for (const auto &p : projector_points)
    out << p << '\n';

  // output the original, shifted and projected points (see the 'for' loop)
  out.close();
  tmp.str("");
  tmp << std::setprecision(8) << "crystal-points-t" << t << ".dat";
  out.open(tmp.str());
  out << "# original shifted projected\n";
#endif

  for (const auto &it : points_surface)
    {
      const auto dp = dp_triple * (p_axis_2 - it.second)[dim - 1] /
                      (p_axis_2 - p_triple)[dim - 1];

      const auto p = it.second + dp;

      points_new[it.first] = surface_projector.project(p);

#ifdef DEBUG
      out << it.second << ' ' << p << ' ' << points_new[it.first] << '\n';
#endif
    }

  for (const auto &it : points_interface)
    {
      const auto dp = calc_interface_displacement(it.second);

      points_new[it.first] = it.second + dp;
    }

  // Update the mesh and obtain displacements for all DoFs.
  // Use improved implementation of function which allows to specify tolerance
  // instead of built-in (from \c GridTools).
  laplace_transform<dim>(points_new,
                         triangulation,
                         nullptr,
                         false,
                         prm.get_double("Laplace tolerance"));

  temperature_solver.get_mesh().clear();
  temperature_solver.get_mesh().copy_triangulation(triangulation);

  temperature_solver.get_support_points(support_points);

  shift.resize(support_points.size());
  for (unsigned int i = 0; i < shift.size(); ++i)
    shift[i] = support_points[i] - support_points_prev[i];

  update_fields();

  // shift the whole mesh according to the pull rate
  GridTools::shift(dz, triangulation);

  // The new triple point is already added, check if the distance is too small
  if ((p_triple_new - p_triple_old).norm() <
      prm.get_double("Surface update tolerance"))
    projector_points.pop_back();

  // shift all points
  for (auto &p : projector_points)
    p += dz;

  surface_projector.set_points(projector_points);

  temperature_solver.get_mesh().clear();
  temperature_solver.get_mesh().copy_triangulation(triangulation);

  if (with_dislocation())
    {
      dislocation_solver.get_mesh().clear();
      dislocation_solver.get_mesh().copy_triangulation(triangulation);
    }

  std::cout << " " << format_time(timer) << "\n";
}

template <int dim>
void
Problem<dim>::calculate_field_gradients()
{
  // evaluate gradients at the previous mesh
  grad_eval.clear();

  grad_eval.attach_dof_handler(temperature_solver.get_dof_handler());
  grad_eval.add_field("T", temperature_solver.get_temperature());

  if (with_dislocation())
    {
      grad_eval.add_field("N_m", dislocation_solver.get_dislocation_density());

      const BlockVector<double> &e_c = dislocation_solver.get_strain_c();

      for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
        grad_eval.add_field("e_c_" + std::to_string(j), e_c.block(j));
    }

  if (f_test.size() > 0)
    grad_eval.add_field("f_test", f_test);

  grad_eval.calculate();

  // output gradients
  const auto         dims = coordinate_names(dim);
  const unsigned int n    = temperature_solver.get_temperature().size();
  Vector<double>     grad_component(n);

  std::vector<std::string> field_names{"T"};

  if (f_test.size() > 0)
    field_names.push_back("f_test");

  // skip gradients of dislocation density solver fields for now

  // initialize displacement with zeros at t=0
  for (unsigned int k = 0; k < dim; ++k)
    temperature_solver.add_field("d" + dims[k], grad_component);

  for (const auto &name : field_names)
    {
      const auto &grad = grad_eval.get_gradient(name);

      for (unsigned int k = 0; k < dim; ++k)
        {
          for (unsigned int i = 0; i < n; ++i)
            grad_component[i] = grad[i][k];

          temperature_solver.add_field("d" + name + "_d" + dims[k],
                                       grad_component);
        }
    }
}

template <int dim>
void
Problem<dim>::calculate_dof_distance()
{
  const DoFHandler<dim> &dh = temperature_solver.get_dof_handler();

  dof_distance.reinit(dh.n_dofs());
  dof_distance.add(std::numeric_limits<double>::max());

  temperature_solver.get_support_points(support_points);

  const unsigned int dofs_per_cell = dh.get_fe().dofs_per_cell;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = dh.begin_active(),
                                                 endc = dh.end();
  for (; cell != endc; ++cell)
    {
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              if (i != j)
                {
                  const double d = (support_points[local_dof_indices[i]] -
                                    support_points[local_dof_indices[j]])
                                     .norm();

                  dof_distance[local_dof_indices[i]] =
                    std::min(dof_distance[local_dof_indices[i]], d);
                  dof_distance[local_dof_indices[j]] =
                    std::min(dof_distance[local_dof_indices[j]], d);
                }
            }
        }
    }

  temperature_solver.add_field("dof_distance", dof_distance);
}

template <int dim>
void
Problem<dim>::update_fields()
{
  Vector<double> &   T = temperature_solver.get_temperature();
  const unsigned int n = T.size();

  const auto &grad_T = grad_eval.get_gradient("T");

  for (unsigned int i = 0; i < n; ++i)
    T[i] += shift[i] * grad_T[i];

  temperature_solver.apply_bc1();

  // output the displacement at DoFs as well
  const auto     dims = coordinate_names(dim);
  Vector<double> shift_component(n);
  for (unsigned int k = 0; k < dim; ++k)
    {
      for (unsigned int i = 0; i < n; ++i)
        shift_component[i] = shift[i][k];

      temperature_solver.add_field("d" + dims[k], shift_component);
    }

  if (f_test.size() > 0)
    {
      const auto &grad_f = grad_eval.get_gradient("f_test");

      for (unsigned int i = 0; i < n; ++i)
        f_test[i] += shift[i] * grad_f[i];

      temperature_solver.add_field("f_test", f_test);
    }

  if (with_dislocation())
    {
      Vector<double> &N_m = dislocation_solver.get_dislocation_density();

      const auto &grad_N_m = grad_eval.get_gradient("N_m");

      for (unsigned int i = 0; i < n; ++i)
        N_m[i] += shift[i] * grad_N_m[i];


      BlockVector<double> &e_c = dislocation_solver.get_strain_c();

      for (unsigned int j = 0; j < e_c.n_blocks(); ++j)
        {
          Vector<double> &e = e_c.block(j);

          const auto &grad_e =
            grad_eval.get_gradient("e_c_" + std::to_string(j));

          for (unsigned int i = 0; i < n; ++i)
            e[i] += shift[i] * grad_e[i];
        }
    }
}

template <int dim>
void
Problem<dim>::initialize_temperature()
{
  temperature_solver.initialize(); // sets T=0

  previous_time_step = temperature_solver.get_time_step();

  Vector<double> &temperature = temperature_solver.get_temperature();
  temperature.add(prm.get_double("Melting point"));

  temperature_solver.output_parameter_table();

  const std::vector<double> X = split_string(prm.get("Probe coordinates x"));
  const std::vector<double> Z = split_string(prm.get("Probe coordinates z"));
  AssertDimension(X.size(), Z.size());

  for (unsigned int i = 0; i < X.size(); ++i)
    {
      Point<dim> p;
      p[0]       = X[i];
      p[dim - 1] = Z[i];
      temperature_solver.add_probe(p);
      dislocation_solver.add_probe(p);
    }

  shift.resize(temperature.size());
  shift_relative.reinit(temperature.size());

  temperature_solver.add_output("V[m/s]", pull_rate->value(Point<1>()));
  temperature_solver.add_output("shift_relative");

  // construct a field for testing
  const std::string expr = prm.get("Interpolation test function");
  if (!expr.empty())
    {
      FunctionParser<2> calc_f;
      calc_f.initialize("r,z", expr, typename FunctionParser<2>::ConstMap());

      temperature_solver.get_support_points(support_points);
      f_test.reinit(temperature.size());

      for (unsigned int i = 0; i < f_test.size(); ++i)
        f_test[i] = calc_f.value(support_points[i]);

      temperature_solver.add_field("f_test", f_test);
    }
}

template <int dim>
void
Problem<dim>::initialize_dislocation()
{
  // initialize dislocations and stresses
  dislocation_solver.initialize();
  dislocation_solver.get_temperature() = temperature_solver.get_temperature();

  dislocation_solver.add_output("V[m/s]", pull_rate->value(Point<1>()));
  dislocation_solver.add_output("shift_relative");

  dislocation_solver.output_parameter_table();

  if (dim == 2)
    dislocation_solver.get_stress_solver().set_bc1(boundary_id_axis, 0, 0);

  update_grid_time_step();

  dislocation_solver.solve(true);

  previous_time_step = temperature_solver.get_time_step();
}

template <int dim>
void
Problem<dim>::set_temperature_BC()
{
  const double T_0 = prm.get_double("Melting point");
  const double e   = prm.get_double("Emissivity");

  temperature_solver.set_bc1(boundary_id_interface, T_0);

  std::function<double(const double)> emissivity_const = [=](const double) {
    return e;
  };

  std::function<double(const double)> emissivity_deriv_const =
    [](const double) { return 0.0; };

  const unsigned int n = temperature_solver.get_temperature().size();
  Vector<double>     q_in(n);

  std::vector<Point<dim>> points;
  std::vector<bool>       boundary_dofs;
  temperature_solver.get_boundary_points(boundary_id_surface,
                                         points,
                                         boundary_dofs);

  for (unsigned int i = 0; i < n; ++i)
    {
      if (!boundary_dofs[i])
        continue;

      const double T = ambient_temperature->value(Point<1>(points[i][dim - 1]));

      q_in[i] = sigma_SB * e * std::pow(T, 4);
    }

  temperature_solver.set_bc_rad_mixed(boundary_id_surface,
                                      q_in,
                                      emissivity_const,
                                      emissivity_deriv_const);
}

template <int dim>
void
Problem<dim>::update_grid_time_step()
{
  const double t  = temperature_solver.get_time();
  const double dt = temperature_solver.get_time_step();
  const double dt_min =
    with_dislocation() ? dislocation_solver.get_time_step_min() : dt;
  const double V = pull_rate->value(Point<1>(t));

  // simple approach based on previous shift and the current pull rate
  const double dL_V = V * dt;

  const unsigned int n = temperature_solver.get_temperature().size();

  AssertDimension(shift.size(), dof_distance.size());
  AssertDimension(shift.size(), n);

  shift_relative.reinit(n);

  for (unsigned int i = 0; i < n; ++i)
    {
      const double dL = shift[i].norm() + dL_V;

      shift_relative[i] = std::max(shift_relative[i], dL / dof_distance[i]);
    }

  temperature_solver.add_field("shift_relative", shift_relative);

  const double relative_shift     = shift_relative.linfty_norm();
  const double max_relative_shift = prm.get_double("Max relative shift");

  temperature_solver.add_output("shift_relative", relative_shift);
  dislocation_solver.add_output("shift_relative", relative_shift);

  const double dt_new =
    dt_min > 0 && relative_shift > 0 && max_relative_shift > 0 ?
      std::max(dt_min,
               dt * std::min(1.0, max_relative_shift / relative_shift)) :
      dt;

  set_time_step(dt_new);

#ifdef DEBUG
  std::cout << "t=" << t << " s relative_shift=" << relative_shift
            << " dt=" << dt << " s dt_new=" << dt_new << " s\n";
#endif
}

template <int dim>
bool
Problem<dim>::update_output_time_step()
{
  const double t  = temperature_solver.get_time();
  const double dt = temperature_solver.get_time_step();

  const double dt_output = prm.get_double("Output time step");

  previous_time_step = dt;

  // output time step is not specified
  if (dt_output <= 0)
    return false;

  // output time too far away, nothing to do
  if (dt_output > 0 && t + (1 + 1e-4) * dt < next_output_time)
    return false;

  // output time is specified and the time step has to be adjusted
  set_time_step(next_output_time - t);

  std::cout << "dt changed from " << dt << " to "
            << temperature_solver.get_time_step()
            << " s for result output at t=" << next_output_time << " s\n";

  next_output_time += dt_output;

  return true;
}

template <int dim>
void
Problem<dim>::set_time_step(const double dt)
{
  temperature_solver.get_time_step() = dt;
  dislocation_solver.get_time_step() = dt;
}

template <int dim>
void
Problem<dim>::solve_steady_temperature()
{
  std::cout << "Calculating steady-state temperature field\n";

  const double dt0 = temperature_solver.get_time_step();

  ParameterHandler &prm_T = temperature_solver.get_parameters();
  // saving as string to avoid ambiguous overloads
  const std::string V0 = prm_T.get("Velocity");

  temperature_solver.get_time_step() = 0;

  // set the pull rate
  prm_T.set("Velocity", pull_rate->value(Point<1>()));

  double max_dT;
  do
    {
      Vector<double> temperature = temperature_solver.get_temperature();

      set_temperature_BC();
      temperature_solver.solve();

      temperature -= temperature_solver.get_temperature();
      max_dT = temperature.linfty_norm();

      std::cout << "max_dT=" << max_dT << " K\n";
  } while (max_dT > prm.get_double("Max temperature change"));

  temperature_solver.get_time_step() = dt0;
  prm_T.set("Velocity", V0);
}

template <int dim>
void
Problem<dim>::solve_temperature()
{
  std::cout << "Calculating transient temperature field\n";

  while (true)
    {
      update_grid_time_step();

      const bool output_enabled = update_output_time_step();

      deform_grid();

      set_temperature_BC();

      const bool keep_going_temp = temperature_solver.solve();

      if (!keep_going_temp)
        break;

      if (output_enabled)
        {
          output_results();

          std::cout << "Restoring previous dt=" << previous_time_step << "s\n";
          set_time_step(previous_time_step);
        }
    };

  output_results();
}

template <int dim>
void
Problem<dim>::solve_temperature_dislocation()
{
  std::cout << "Calculating temperature and dislocation density\n";

  while (true)
    {
      update_grid_time_step();

      const bool output_enabled = update_output_time_step();

      set_time_step(dislocation_solver.get_time_step());

      deform_grid();

      set_temperature_BC();

      const bool keep_going_temp = temperature_solver.solve();

      dislocation_solver.get_temperature() =
        temperature_solver.get_temperature();

      const bool keep_going_disl = dislocation_solver.solve();

      if (!keep_going_temp || !keep_going_disl)
        break;

      if (output_enabled)
        {
          output_results();

          std::cout << "Restoring previous dt=" << previous_time_step << "s\n";
          set_time_step(previous_time_step);
        }
    };

  output_results();
}

template <int dim>
void
Problem<dim>::output_results(const bool data,
                             const bool vtk,
                             const bool boundary,
                             const bool mesh) const
{
  const bool has_dislocation =
    with_dislocation() &&
    dislocation_solver.get_dof_handler().has_active_dofs();

  if (data)
    {
      temperature_solver.output_data();
      if (has_dislocation)
        dislocation_solver.output_data();
    }

  if (vtk)
    {
      temperature_solver.output_vtk();
      if (has_dislocation)
        dislocation_solver.output_vtk();
    }

  // Exports values at all the boundaries in 2D.
  // Export is disabled by default in 3D.
  if (boundary)
    {
      temperature_solver.output_boundary_values(boundary_id_interface);
      temperature_solver.output_boundary_values(boundary_id_surface);
      temperature_solver.output_boundary_values(boundary_id_axis);
      if (has_dislocation)
        {
          dislocation_solver.output_boundary_values(boundary_id_interface);
          dislocation_solver.output_boundary_values(boundary_id_surface);
          dislocation_solver.output_boundary_values(boundary_id_axis);
        }
    }

  if (mesh)
    {
      temperature_solver.output_mesh();
    }
}

template <int dim>
bool
Problem<dim>::with_dislocation() const
{
  return !prm.get_bool("Temperature only");
}

int
main(int argc, char *argv[])
{
  const std::vector<std::string> arguments(argv, argv + argc);

  bool init      = false;
  int  order     = 2;
  int  dimension = 2;

  for (unsigned int i = 1; i < arguments.size(); ++i)
    {
      if (arguments[i] == "init" || arguments[i] == "use_default_prm")
        init = true;
      if (arguments[i] == "order" && i + 1 < arguments.size())
        order = std::stoi(arguments[i + 1]);
    }

  deallog.attach(std::cout);
  deallog.depth_console(2);

  if (dimension == 2)
    {
      Problem<2> p2d(order, init);
      if (!init)
        p2d.run();
    }
  else
    {
      throw ExcNotImplemented();
    }

  return 0;
}
