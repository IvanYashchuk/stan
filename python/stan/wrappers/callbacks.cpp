#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>

#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define BIND_LOGGER_METHOD(name, type) \
    void name(type message) override { \
      PYBIND11_OVERRIDE_PURE( \
        void, \
        stan::callbacks::logger, \
        name, \
        message \
      ); \
    }

namespace stan_wrappers
{
  class py_interrupt : public stan::callbacks::interrupt {
  public:
    /* Trampoline (need one for each virtual function) */
    void operator()() override {
      PYBIND11_OVERRIDE_PURE_NAME(
        void, /* Return type */
        stan::callbacks::interrupt,      /* Parent class */
        "__call__", /* Name of function in Python */
        operator()          /* Name of function in C++ */
      );
    }
  };

  class py_logger : public stan::callbacks::logger {
  public:
    /* Trampoline (need one for each virtual function) */
    // void debug(const std::string& message) override {
    //   PYBIND11_OVERRIDE_PURE(
    //     void,  /* Return type */
    //     stan::callbacks::interrupt,  /* Parent class */
    //     debug,  /* Name of function in C++ (must match Python name) */
    //     message  /* Argument(s) */
    //   );
    // }
    BIND_LOGGER_METHOD(debug, const std::string&);
    BIND_LOGGER_METHOD(debug, const std::stringstream&);

    BIND_LOGGER_METHOD(info, const std::string&);
    BIND_LOGGER_METHOD(info, const std::stringstream&);

    BIND_LOGGER_METHOD(warn, const std::string&);
    BIND_LOGGER_METHOD(warn, const std::stringstream&);

    BIND_LOGGER_METHOD(error, const std::string&);
    BIND_LOGGER_METHOD(error, const std::stringstream&);

    BIND_LOGGER_METHOD(fatal, const std::string&);
    BIND_LOGGER_METHOD(fatal, const std::stringstream&);
  };

  class py_writer : public stan::callbacks::writer {
  public:
    /* Trampoline (need one for each virtual function) */
    void operator()(const std::vector<std::string>& names) override {
      PYBIND11_OVERRIDE_PURE_NAME(
        void, /* Return type */
        stan::callbacks::writer,  /* Parent class */
        "__call__",  /* Name of function in Python */
        operator(),  /* Name of function in C++ */
        names,  /* Argument(s) */
      );
    }

    void operator()(const std::vector<double>& state) override {
      PYBIND11_OVERRIDE_PURE_NAME(
        void, /* Return type */
        stan::callbacks::writer,  /* Parent class */
        "__call__",  /* Name of function in Python */
        operator(),  /* Name of function in C++ */
        names,  /* Argument(s) */
      );
    }

    void operator()() override {
      PYBIND11_OVERRIDE_PURE_NAME(
        void, /* Return type */
        stan::callbacks::writer,  /* Parent class */
        "__call__",  /* Name of function in Python */
        operator()  /* Name of function in C++ */
      );
    }

    void operator()(const std::string& message) override {
      PYBIND11_OVERRIDE_PURE_NAME(
        void, /* Return type */
        stan::callbacks::writer,  /* Parent class */
        "__call__",  /* Name of function in Python */
        operator(),  /* Name of function in C++ */
        message,  /* Argument(s) */
      );
    }
  };

  void callbacks(py::module& m) {
  // stan::callbacks::interrupt
  py::class_<stan::callbacks::interrupt, py_interrupt>(
    m, "interrupt", "interrupt class")
    .def(py::init<>())
    .def("__call__", &stan::callbacks::interrupt::operator(), "Callback function");

  // stan::callbacks::logger
  py::class_<stan::callbacks::logger, py_logger>(
    m, "logger", "logger class")
    .def(py::init<>())
    .def("debug", &stan::callbacks::logger::debug, "Logs a message with debug log level")
    .def("info", &stan::callbacks::logger::info, "Logs a message with info log level")
    .def("warn", &stan::callbacks::logger::warn, "Logs a message with warn log level")
    .def("error", &stan::callbacks::logger::error, "Logs a message with error log level")
    .def("fatal", &stan::callbacks::logger::fatal, "Logs a message with fatal log level");

  // stan::callbacks::writer
  py::class_<stan::callbacks::writer, py_writer>(
    m, "writer", "writer class")
    .def(py::init<>())
    .def("__call__", &stan::callbacks::writer::operator(), "Writes a set of names/set of values/blank input/string");
  }
}  // namespace stan_wrappers
