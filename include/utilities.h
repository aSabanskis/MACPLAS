#ifndef macplas_utilities_h
#define macplas_utilities_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

using namespace dealii;

// helper functions
double
sqr(const double x);

template <int dim>
Point<dim>
closest_segment_point(const Point<dim> &p,
                      const Point<dim> &segment_p0,
                      const Point<dim> &segment_p1);

// helper classe
template <int dim>
class Triangle
{
public:
  void
  reinit(const Point<dim> &p0, const Point<dim> &p1, const Point<dim> &p2);

  Point<dim>
  center() const;

  Point<dim>
  normal() const;

  double
  area() const;

  double
  longest_side() const;

  Point<dim>
  closest_triangle_point(const Point<dim> &p) const;

  std::array<double, 3>
  barycentric_coordinates(const Point<dim> &p) const;

private:
  void
  calculate_normal_and_area();

  double
  signed_area(const Point<dim> &p0,
              const Point<dim> &p1,
              const Point<dim> &p2) const;

  Point<dim>
  project_to_triangle_plane(const Point<dim> &p) const;

  std::array<Point<dim>, 3> m_points;
  Point<dim>                m_normal;
  Point<dim>                m_center;
  double                    m_area;
  double                    m_longest_side;
};


// Class for interpolation of boundary conditions from external cell or point
// data defined on a triangulated surface.
class SurfaceInterpolator3D
{
private:
  constexpr static unsigned int dim = 3; // everything is in 3D

public:
  enum FieldType
  {
    CellField,
    PointField
  };

  // read mesh and fields from vtk file
  void
  read_vtk(const std::string &file_name);

  // read mesh and fields from vtu file
  void
  read_vtu(const std::string &file_name);

  // write mesh and fields to vtu file
  void
  write_vtu(const std::string &file_name) const;

  // interpolate cell field to the specified points
  void
  interpolate(const FieldType &              field_type,
              const std::string &            field_name,
              const std::vector<Point<dim>> &target_points,
              const std::vector<bool> &      markers,
              Vector<double> &               target_values) const;

  // convert between cell and point fields
  // If target_name is not specified it is set to source_name.
  void
  convert(const FieldType &  source_type,
          const std::string &source_name,
          const FieldType &  target_type,
          const std::string &target_name = "");

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

  // precalculated triangle areas, normals, centers etc.
  std::vector<Triangle<dim>> triangle_cache;


  // get field
  const std::vector<double> &
  field(const FieldType &field_type, const std::string &field_name) const;
  std::vector<double> &
  field(const FieldType &field_type, const std::string &field_name);
  const std::vector<Point<dim>> &
  vector_field(const FieldType &  field_type,
               const std::string &field_name) const;

  // convert field
  void
  cell_to_point(const std::string &source_name, const std::string &target_name);
  void
  point_to_cell(const std::string &source_name, const std::string &target_name);

  // clear all data
  void
  clear();

  // print mesh and field information
  void
  info() const;

  // compute auxiliary data (cell areas, centers, normals)
  void
  preprocess();
};


// IMPLEMENTATION

double
sqr(const double x)
{
  return x * x;
}

template <int dim>
Point<dim>
closest_segment_point(const Point<dim> &p,
                      const Point<dim> &segment_p0,
                      const Point<dim> &segment_p1)
{
  const auto d = segment_p1 - segment_p0;
  double     t = (d * (p - segment_p0)) / d.norm();
  // clamp to [0,1]
  t = std::max(0.0, std::min(1.0, t));

  return segment_p0 + t * d;
}

// Triangle

template <int dim>
void
Triangle<dim>::reinit(const Point<dim> &p0,
                      const Point<dim> &p1,
                      const Point<dim> &p2)
{
  m_points[0] = p0;
  m_points[1] = p1;
  m_points[2] = p2;

  calculate_normal_and_area();

  m_center = (p0 + p1 + p2) / 3;

  m_longest_side =
    std::max(std::max((p1 - p0).norm(), (p2 - p0).norm()), (p1 - p2).norm());
}

template <int dim>
Point<dim>
Triangle<dim>::center() const
{
  return m_center;
}

template <int dim>
Point<dim>
Triangle<dim>::normal() const
{
  return m_normal;
}

template <int dim>
double
Triangle<dim>::area() const
{
  return m_area;
}

template <int dim>
double
Triangle<dim>::longest_side() const
{
  return m_longest_side;
}

template <int dim>
Point<dim>
Triangle<dim>::closest_triangle_point(const Point<dim> &p) const
{
  const Point<dim> p_proj = project_to_triangle_plane(p);

  const auto t3 = barycentric_coordinates(p_proj);

  bool inside = true;

  for (const auto &t : t3)
    inside = inside && (t >= 0) && (t <= 1);

  if (inside)
    {
      return p_proj;
    }
  else
    {
      Point<dim> p_closest;
      double     d2_min = -1;

      for (unsigned int i = 0; i < 3; ++i)
        {
          const Point<dim> p_edge =
            closest_segment_point(p, m_points[i], m_points[(i + 1) % 3]);
          const double d2 = (p - p_edge).norm_square();

          if (d2 < d2_min || d2_min < 0)
            {
              d2_min    = d2;
              p_closest = p_edge;
            }
        }

      return p_closest;
    }
}

template <int dim>
void
Triangle<dim>::calculate_normal_and_area()
{
  m_normal = Point<dim>(
    cross_product_3d(m_points[1] - m_points[0], m_points[2] - m_points[0]));

  m_area = 0.5 * m_normal.norm();

  // normalize
  if (m_area > 0)
    m_normal /= (2 * m_area);
}

template <int dim>
double
Triangle<dim>::signed_area(const Point<dim> &p0,
                           const Point<dim> &p1,
                           const Point<dim> &p2) const
{
  return 0.5 * (m_normal * cross_product_3d(p1 - p0, p2 - p0));
}

template <int dim>
Point<dim>
Triangle<dim>::project_to_triangle_plane(const Point<dim> &p) const
{
  return Point<dim>(p - m_normal * (m_normal * (p - m_points[0])));
}

template <int dim>
std::array<double, 3>
Triangle<dim>::barycentric_coordinates(const Point<dim> &p) const
{
  return std::array<double, 3>(
    {signed_area(p, m_points[1], m_points[2]) / m_area,
     signed_area(m_points[0], p, m_points[2]) / m_area,
     signed_area(m_points[0], m_points[1], p) / m_area});
}

// SurfaceInterpolator3D

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
SurfaceInterpolator3D::write_vtu(const std::string &file_name) const
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
SurfaceInterpolator3D::interpolate(const FieldType &              field_type,
                                   const std::string &            field_name,
                                   const std::vector<Point<dim>> &target_points,
                                   const std::vector<bool> &      markers,
                                   Vector<double> &target_values) const
{
  AssertThrow(field_type == CellField || field_type == PointField,
              ExcNotImplemented());

  const std::vector<double> &source_field = field(field_type, field_name);

  const unsigned int n_triangles = triangles.size();
  const unsigned int n_values    = target_points.size();
  target_values.reinit(n_values);

  Triangle<dim> triangle;

  for (unsigned int i = 0; i < n_values; ++i)
    {
      target_values[i] = 0;

      if (!markers[i])
        continue;

      Point<dim>   p_found;
      unsigned int j_found = 0;
      double       d2_min  = -1;
      for (unsigned int j = 1; j < n_triangles; ++j)
        {
          const Triangle<dim> &triangle = triangle_cache[j];

          if ((target_points[i] - triangle.center()).norm() >
              3 * triangle.longest_side())
            continue;

          Point<dim> p_trial =
            triangle.closest_triangle_point(target_points[i]);

          double d2 = (p_trial - target_points[i]).norm_square();
          if (d2 < d2_min || d2_min < 0)
            {
              d2_min  = d2;
              p_found = p_trial;
              j_found = j;
            }
        }

      switch (field_type)
        {
          case CellField:
            target_values[i] = source_field[j_found];
            break;

          case PointField:
            const Triangle<dim> &triangle = triangle_cache[j_found];
            const auto           t3 = triangle.barycentric_coordinates(p_found);
            const auto &         v  = triangles[j_found];
            for (unsigned int k = 0; k < 3; ++k)
              target_values[i] += t3[k] * source_field[v[k]];
            break;
        }
    }
}

void
SurfaceInterpolator3D::convert(const FieldType &  source_type,
                               const std::string &source_name,
                               const FieldType &  target_type,
                               const std::string &target_name)
{
  const std::string target_name_updated =
    target_name.empty() ? source_name : target_name;

  if (source_type == CellField && target_type == PointField)
    {
      cell_to_point(source_name, target_name_updated);
    }
  else if (source_type == PointField && target_type == CellField)
    {
      point_to_cell(source_name, target_name_updated);
    }
  else
    {
      throw std::runtime_error(
        "Unsupported combination of source and target field types.");
    }
}

const std::vector<double> &
SurfaceInterpolator3D::field(const FieldType &  field_type,
                             const std::string &field_name) const
{
  const auto &fields = field_type == CellField ? cell_fields : point_fields;
  const auto &it     = fields.find(field_name);

  if (it == fields.end())
    throw std::runtime_error("Field '" + field_name + "' does not exist.");

  return it->second;
}

std::vector<double> &
SurfaceInterpolator3D::field(const FieldType &  field_type,
                             const std::string &field_name)
{
  return field_type == CellField ? cell_fields[field_name] :
                                   point_fields[field_name];
}

const std::vector<Point<SurfaceInterpolator3D::dim>> &
SurfaceInterpolator3D::vector_field(const FieldType &  field_type,
                                    const std::string &field_name) const
{
  AssertThrow(field_type == CellField, ExcNotImplemented());

  const auto &fields = cell_vector_fields;
  const auto &it     = fields.find(field_name);

  if (it == fields.end())
    throw std::runtime_error("Field '" + field_name + "' does not exist.");

  return it->second;
}

void
SurfaceInterpolator3D::cell_to_point(const std::string &source_name,
                                     const std::string &target_name)
{
  const unsigned int n_points    = points.size();
  const unsigned int n_triangles = triangles.size();

  const std::vector<double> &source_field = field(CellField, source_name);

  std::vector<double> &target_field = field(PointField, target_name);
  target_field.resize(n_points);
  std::fill(target_field.begin(), target_field.end(), 0);

  std::vector<int> count(n_points, 0);

  for (unsigned int i = 0; i < n_triangles; ++i)
    {
      const auto &v = triangles[i];
      for (const auto &id : v)
        {
          target_field[id] += source_field[i];
          count[id]++;
        }
    }

  for (unsigned int i = 0; i < n_points; ++i)
    {
      if (count[i] > 0)
        target_field[i] /= count[i];
    }
}

void
SurfaceInterpolator3D::point_to_cell(const std::string &source_name,
                                     const std::string &target_name)
{
  const unsigned int n_points    = points.size();
  const unsigned int n_triangles = triangles.size();

  const std::vector<double> &source_field = field(PointField, source_name);

  std::vector<double> &target_field = field(CellField, target_name);
  target_field.resize(n_triangles);
  std::fill(target_field.begin(), target_field.end(), 0);

  std::vector<int> count(n_triangles, 0);

  for (unsigned int i = 0; i < n_triangles; ++i)
    {
      const auto &v = triangles[i];
      for (const auto &id : v)
        {
          target_field[i] += source_field[id];
          count[i]++;
        }
    }

  for (unsigned int i = 0; i < n_triangles; ++i)
    {
      if (count[i] > 0)
        target_field[i] /= count[i];
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
  triangle_cache.clear();
}

void
SurfaceInterpolator3D::info() const
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

  auto &area         = cell_fields["area"];
  auto &longest_side = cell_fields["longest_side"];
  auto &center       = cell_vector_fields["center"];
  auto &normal       = cell_vector_fields["normal"];

  area.resize(n_triangles);
  longest_side.resize(n_triangles);
  center.resize(n_triangles);
  normal.resize(n_triangles);
  triangle_cache.resize(n_triangles);

  for (unsigned int i = 0; i < n_triangles; ++i)
    {
      const auto &   v        = triangles[i];
      Triangle<dim> &triangle = triangle_cache[i];
      triangle.reinit(points[v[0]], points[v[1]], points[v[2]]);

      // for output, should be moved to debug mode
      center[i]       = triangle.center();
      normal[i]       = triangle.normal();
      area[i]         = triangle.area();
      longest_side[i] = triangle.longest_side();
    }
}

#endif