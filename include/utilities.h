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

class Interpolator3D
{
public:
  void
  read_vtu(const std::string &file_name);

  void
  write_vtu(const std::string &file_name);

private:
  constexpr static unsigned int dim = 3; // everything is in 3D

  // list of points (x,y,z)
  std::vector<Point<dim>> points;

  // connectivity matrix (v0,v1,v2)
  std::vector<std::array<unsigned int, 3>> triangles;

  // fields defined on cells
  std::map<std::string, std::vector<double>> cell_fields;

  // fields defined on points
  std::map<std::string, std::vector<double>> point_fields;

  // vector fields defined on cells
  std::map<std::string, std::vector<Point<dim>>> cell_vector_fields;

  void
  preprocess();
};

void
Interpolator3D::read_vtu(const std::string &file_name)
{
  points.clear();
  triangles.clear();
  cell_fields.clear();
  point_fields.clear();
  cell_vector_fields.clear();

  std::ifstream vtk(file_name);

  if (!vtk.is_open())
    return;

  std::string s, data_type, data_name;

  bool data_start = false;

  while (vtk >> s)
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

          while (vtk >> s)
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

  std::cout << "n_points:" << points.size() << " "
            << "n_triangles:" << triangles.size() << "\n";

  for (const auto &it : cell_fields)
    std::cout << "CellData " << it.first << " " << it.second.size() << " "
              << it.second.back() << "\n";

  for (const auto &it : point_fields)
    std::cout << "PointData " << it.first << " " << it.second.size() << " "
              << it.second.back() << "\n";

  preprocess();
}

void
Interpolator3D::write_vtu(const std::string &file_name)
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
Interpolator3D::preprocess()
{
  const unsigned int n_points    = points.size();
  const unsigned int n_triangles = triangles.size();

  auto &area   = cell_fields["area"];
  auto &normal = cell_vector_fields["normal"];

  normal.resize(n_triangles);
  area.resize(n_triangles);

  for (unsigned int i = 0; i < n_triangles; ++i)
    {
      const auto &v     = triangles[i];
      const auto  a     = points[v[1]] - points[v[0]];
      const auto  b     = points[v[2]] - points[v[0]];
      const auto  a_x_b = cross_product_3d(a, b);
      const auto  norm  = a_x_b.norm();

      area[i]   = 0.5 * norm;
      normal[i] = Point<dim>(a_x_b / norm);
    }
}

#endif