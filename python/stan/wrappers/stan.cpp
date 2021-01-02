#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace stan_wrappers
{
  void callbacks(py::module& m);
  void model(py::module& m);
} // namespace stan_wrappers

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "Stan Python interface";
  m.attr("__version__") = "0.1";

  // Create callbacks submodule [callbacks]
  py::module callbacks = m.def_submodule("callbacks", "Callbacks module");
  stan_wrappers::callbacks(callbacks);

  // Create model submodule [model]
  py::module model = m.def_submodule("model", "Model module");
  stan_wrappers::model(model);
}