#ifndef macplas_utilities_h
#define macplas_utilities_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

using namespace dealii;

// Class for interpolation of boundary conditions from external cell or point
// data defined on a triangulated surface.
class SurfaceInterpolator3D
{
private:
  constexpr static unsigned int dim = 3; // everything is in 3D

public:
  // read mesh and fields from vtk file
  void
  read_vtk(const std::string &file_name);

  // read mesh and fields from vtu file
  void
  read_vtu(const std::string &file_name);

  // write mesh and fields to vtu file
  void
  write_vtu(const std::string &file_name);

  // interpolate cell field to the specified points
  void
  interpolate_cell(const std::string &            field_name,
                   const std::vector<Point<dim>> &target_points,
                   const std::vector<bool> &      markers,
                   Vector<double> &               target_values);

private:
  // list of points (x,y,z)
  std::vector<Point<dim>> points;

  // connectivity matrix (v0,v1,v2) - only triangles
  std::vector<std::array<unsigned int, 3>> triangles;

  // fields defined on cells
  std::map<std::string, std::vector<double>> cell_fields;

  // fields defined on points
  std::map<std::string, std::vector<double>> point_fields;

  // vector fields defined on cells
  std::map<std::string, std::vector<Point<dim>>> cell_vector_fields;

  // clear all data
  void
  clear();

  // print mesh and field information
  void
  info();

  // compute auxiliary data (cell areas, centers, normals)
  void
  preprocess();
};


void
SurfaceInterpolator3D::read_vtk(const std::string &file_name)
{
  clear();

  std::ifstream file(file_name);

  if (!file.is_open())
    {
      std::cout << "Could not open '" << file_name << "'\n";
      return;
    }

  std::string s, data_type, data_name;

  // First line should be "# vtk DataFile Version 4.2"
  while (file >> s)
    {
      if (s == "POINTS")
        {
          int n;
          file >> n >> s /*data type*/;
          points.resize(n);

          for (unsigned int i = 0; i < n; ++i)
            {
              file >> points[i][0] >> points[i][1] >> points[i][2];
            }
        }
      else if (s == "CELLS")
        {
          int n;
          file >> n >> s /*size*/;
          triangles.resize(n);

          for (unsigned int i = 0; i < n; ++i)
            {
              file >> s;
              if (s != "3")
                throw std::runtime_error("Triangle expected, numPoints=" + s +
                                         "found");
              file >> triangles[i][0] >> triangles[i][1] >> triangles[i][2];
            }
        }
      else
        {
          if (s == "CELL_DATA" || s == "POINT_DATA")
            data_type = s;

          if (s == "SCALARS")
            {
              file >> data_name >> s /*data type*/;
              file >> s >> s; // LOOKUP_TABLE

              unsigned int N =
                data_type == "CELL_DATA" ? triangles.size() : points.size();

              std::cout << data_type << " " << data_name << " " << N << "\n";

              std::vector<double> &f = data_type == "CELL_DATA" ?
                                         cell_fields[data_name] :
                                         point_fields[data_name];
              f.resize(N);

              for (unsigned int i = 0; i < N; ++i)
                file >> f[i];
            }

          if (s == "FIELD")
            {
              int n_fields;
              file >> s /*FieldData*/ >> n_fields;

              for (unsigned int k = 0; k < n_fields; ++k)
                {
                  file >> data_name >> s >> s >> s;

                  unsigned int N =
                    data_type == "CELL_DATA" ? triangles.size() : points.size();

                  std::cout << data_type << " " << data_name << " " << N
                            << "\n";

                  std::vector<double> &f = data_type == "CELL_DATA" ?
                                             cell_fields[data_name] :
                                             point_fields[data_name];
                  f.resize(N);

                  for (unsigned int i = 0; i < N; ++i)
                    file >> f[i];
                }
            }
        }
    }

  info();
  preprocess();
}

void
SurfaceInterpolator3D::read_vtu(const std::string &file_name)
{
  clear();

  std::ifstream file(file_name);

  if (!file.is_open())
    {
      std::cout << "Could not open '" << file_name << "'\n";
      return;
    }

  std::string s, data_type, data_name;

  bool data_start = false;

  while (file >> s)
    {
      // get number of points and cells
      if (Utilities::match_at_string_start(s, "NumberOf"))
        {
          std::vector<std::string> l = Utilities::split_string_list(s, '"');

          int n = Utilities::string_to_int(l[1]);

          if (l[0] == "NumberOfPoints=")
            points.resize(n);
          else if (l[0] == "NumberOfCells=")
            triangles.resize(n);

          continue;
        }

      // detect cell and point data (also points and cells)
      if (Utilities::match_at_string_start(s, "<Cell") ||
          Utilities::match_at_string_start(s, "<Point"))
        data_type = s;

      // get name
      if (Utilities::match_at_string_start(s, "Name="))
        data_name = Utilities::split_string_list(s, '"')[1];

      // prepare for reading data
      if (s == "<DataArray")
        data_start = true;

      // <DataArray ...> has been reached, now read the data
      if (data_start && s.back() == '>')
        {
          std::cout << data_type << " " << data_name << "\n";

          std::vector<std::string> data;

          while (file >> s)
            {
              // end of data array reached, process it
              if (s == "</DataArray>")
                {
                  const unsigned int n = data.size();

                  if (data_type == "<CellData>") // cell field
                    {
                      if (n != triangles.size())
                        throw std::runtime_error(
                          data_type + " " + std::to_string(triangles.size()) +
                          " " + data_name + " " + std::to_string(n));

                      auto &field = cell_fields[data_name];
                      field.resize(n);

                      for (unsigned int i = 0; i < n; ++i)
                        field[i] = std::stod(data[i]);
                    }
                  else if (data_type == "<PointData>") // point field
                    {
                      if (n != points.size())
                        throw std::runtime_error(
                          data_type + " " + std::to_string(points.size()) +
                          " " + data_name + " " + std::to_string(n));

                      auto &field = point_fields[data_name];
                      field.resize(n);

                      for (unsigned int i = 0; i < n; ++i)
                        field[i] = std::stod(data[i]);
                    }
                  else if (data_type == "<Points>") // point coordinates
                    {
                      if (n != 3 * points.size())
                        throw std::runtime_error(
                          data_type + " " + std::to_string(points.size()) +
                          " " + data_name + " " + std::to_string(n));

                      for (unsigned int i = 0; i < n / 3; ++i)
                        {
                          unsigned int k = 3 * i;

                          points[i] = Point<dim>(std::stod(data[k]),
                                                 std::stod(data[k + 1]),
                                                 std::stod(data[k + 2]));
                        }
                    }
                  else if (data_type == "<Cells>" &&
                           data_name == "connectivity") // only connectivity
                    {
                      if (n != 3 * triangles.size())
                        throw std::runtime_error(
                          data_type + " " + std::to_string(points.size()) +
                          " " + data_name + " " + std::to_string(n));

                      for (unsigned int i = 0; i < n / 3; ++i)
                        {
                          unsigned int k = 3 * i;

                          triangles[i] = {std::stoi(data[k]),
                                          std::stoi(data[k + 1]),
                                          std::stoi(data[k + 2])};
                        }
                    }

                  data_start = false;
                  data_name.clear();
                  break;
                }
              else // append
                data.push_back(s);
            }
        }
    }

  info();
  preprocess();
}

void
SurfaceInterpolator3D::write_vtu(const std::string &file_name)
{
  std::ofstream f_out(file_name);

  const unsigned int n_points    = points.size();
  const unsigned int n_triangles = triangles.size();

  f_out
    << "<?xml version=\"1.0\"?>\n"
       "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
       "<UnstructuredGrid>\n"
       "<Piece NumberOfPoints=\""
    << n_points << "\" NumberOfCells=\"" << n_triangles << "\">\n";


  f_out << "<CellData>\n";
  for (const auto &it : cell_fields)
    {
      f_out << "<DataArray type=\"Float64\" Name=\"" << it.first
            << "\" format=\"ascii\">\n";
      for (const auto &x : it.second)
        f_out << std::scientific << std::setprecision(14) << x << " ";
      f_out << "\n</DataArray>\n";
    }
  for (const auto &it : cell_vector_fields)
    {
      f_out << "<DataArray type=\"Float64\" Name=\"" << it.first
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (const auto &x : it.second)
        f_out << std::scientific << std::setprecision(14) << x << "\n";
      f_out << "</DataArray>\n";
    }
  f_out << "</CellData>\n";


  f_out << "<PointData>\n";
  for (const auto &it : point_fields)
    {
      f_out << "<DataArray type=\"Float64\" Name=\"" << it.first
            << "\" format=\"ascii\">\n";
      for (const auto &x : it.second)
        f_out << std::scientific << std::setprecision(14) << x << " ";
      f_out << "\n</DataArray>\n";
    }
  f_out << "</PointData>\n";


  f_out << "<Points>\n";
  f_out
    << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (const auto &p : points)
    f_out << std::scientific << std::setprecision(14) << p << "\n";
  f_out << "</DataArray>\n";
  f_out << "</Points>\n";


  f_out << "<Cells>\n";

  f_out
    << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (const auto &t : triangles)
    {
      for (const auto &v : t)
        f_out << v << " ";
      f_out << "\n";
    }
  f_out << "</DataArray>\n";

  f_out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (unsigned int i = 0; i < n_triangles; ++i)
    f_out << 3 * (i + 1) << " ";
  f_out << "\n</DataArray>\n";

  f_out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (unsigned int i = 0; i < n_triangles; ++i)
    f_out << "5 "; // VTK_TRIANGLE
  f_out << "\n</DataArray>\n";

  f_out << "</Cells>\n";


  f_out << "</Piece>\n"
           "</UnstructuredGrid>\n"
           "</VTKFile>\n";
}

void
SurfaceInterpolator3D::interpolate_cell(
  const std::string &            field_name,
  const std::vector<Point<dim>> &target_points,
  const std::vector<bool> &      markers,
  Vector<double> &               target_values)
{
  const auto &it = cell_fields.find(field_name);
  if (it == cell_fields.end())
    throw std::runtime_error("Field '" + field_name + "' does not exist.");

  const std::vector<double> &    f = it->second;
  const std::vector<Point<dim>> &c = cell_vector_fields["center"];

  const unsigned int n_triangles = triangles.size();
  const unsigned int n_values    = target_points.size();
  target_values.reinit(n_values);

  for (unsigned int i = 0; i < n_values; ++i)
    {
      if (!markers[i])
        continue;

      // for now, find the closest triangle midpoint
      unsigned int j_min = 0;
      double       d_min = c[j_min].distance(target_points[i]);
      for (unsigned int j = 1; j < n_triangles; ++j)
        {
          double d = c[j].distance(target_points[i]);
          if (d < d_min)
            {
              d_min = d;
              j_min = j;
            }
        }

      target_values[i] = f[j_min];
    }
}

void
SurfaceInterpolator3D::clear()
{
  points.clear();
  triangles.clear();
  cell_fields.clear();
  point_fields.clear();
  cell_vector_fields.clear();
}

void
SurfaceInterpolator3D::info()
{
  std::cout << "n_points:" << points.size() << " "
            << "n_triangles:" << triangles.size() << "\n";

  for (const auto &it : cell_fields)
    std::cout << "CellData " << it.first << " " << it.second.size() << "\n";

  for (const auto &it : point_fields)
    std::cout << "PointData " << it.first << " " << it.second.size() << "\n";
}

void
SurfaceInterpolator3D::preprocess()
{
  const unsigned int n_triangles = triangles.size();

  auto &area   = cell_fields["area"];
  auto &center = cell_vector_fields["center"];
  auto &normal = cell_vector_fields["normal"];

  area.resize(n_triangles);
  center.resize(n_triangles);
  normal.resize(n_triangles);

  for (unsigned int i = 0; i < n_triangles; ++i)
    {
      const auto &v     = triangles[i];
      const auto  a     = points[v[1]] - points[v[0]];
      const auto  b     = points[v[2]] - points[v[0]];
      const auto  a_x_b = cross_product_3d(a, b);
      const auto  norm  = a_x_b.norm();

      area[i]   = 0.5 * norm;
      center[i] = (points[v[0]] + points[v[1]] + points[v[2]]) / 3;
      normal[i] = norm > 0 ? Point<dim>(a_x_b / norm) : Point<dim>();
    }
}

#endif