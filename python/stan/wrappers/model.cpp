#include <stan/model/model_base.hpp>

#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace stan_wrappers
{
  class py_model_base : public stan::model::model_base {
  public:
    /* Inherit the constructors */
    using stan::model::model_base::model_base;

    /* Trampoline (need one for each virtual function) */
    std::string model_name() override {
      PYBIND11_OVERRIDE_PURE(
        std::string, /* Return type */
        stan::model::model_base,  /* Parent class */
        model_name  /* Name of function in C++ */
      );
    }

    std::vector<std::string> model_compile_info() override {
      PYBIND11_OVERRIDE_PURE(
        std::vector<std::string>, /* Return type */
        stan::model::model_base,  /* Parent class */
        model_compile_info  /* Name of function in C++ */
      );
    }

    void get_param_names(std::vector<std::string>& names) override {
      PYBIND11_OVERRIDE_PURE(
        void, /* Return type */
        stan::model::model_base,  /* Parent class */
        model_compile_info,  /* Name of function in C++ */
        names
      );
    }

    void get_dims(std::vector<std::vector<size_t> >& dimss) override {
      PYBIND11_OVERRIDE_PURE(
        void, /* Return type */
        stan::model::model_base,  /* Parent class */
        get_dims,  /* Name of function in C++ */
        dimss
      );
    }

    void constrained_param_names(std::vector<std::string>& param_names, bool include_tparams = true, bool include_gqs = true) override {
      PYBIND11_OVERRIDE_PURE(
        void, /* Return type */
        stan::model::model_base,  /* Parent class */
        constrained_param_names,  /* Name of function in C++ */
        param_names, include_tparams, include_gqs
      );
    }

    void unconstrained_param_names(std::vector<std::string>& param_names, bool include_tparams = true, bool include_gqs = true) override {
      PYBIND11_OVERRIDE_PURE(
        void, /* Return type */
        stan::model::model_base,  /* Parent class */
        unconstrained_param_names,  /* Name of function in C++ */
        param_names, include_tparams, include_gqs
      );
    }

    double log_prob(Eigen::VectorXd& params_r, std::ostream* msgs) override {
      PYBIND11_OVERRIDE_PURE(
        double, /* Return type */
        stan::model::model_base,  /* Parent class */
        log_prob,  /* Name of function in C++ */
        params_r, msgs
      );
    }

    // stan::math::var log_prob(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) override {
    //   PYBIND11_OVERRIDE_PURE(
    //     stan::math::var, /* Return type */
    //     stan::model::model_base,  /* Parent class */
    //     log_prob,  /* Name of function in C++ */
    //     params_r, msgs
    //   );
    // }

    double log_prob_jacobian(Eigen::VectorXd& params_r, std::ostream* msgs) override {
      PYBIND11_OVERRIDE_PURE(
        double, /* Return type */
        stan::model::model_base,  /* Parent class */
        log_prob_jacobian,  /* Name of function in C++ */
        params_r, msgs
      );
    }

    // stan::math::var log_prob_jacobian(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) override {
    //   PYBIND11_OVERRIDE_PURE(
    //     stan::math::var, /* Return type */
    //     stan::model::model_base,  /* Parent class */
    //     log_prob_jacobian,  /* Name of function in C++ */
    //     params_r, msgs
    //   );
    // }

    double log_prob_propto(Eigen::VectorXd& params_r, std::ostream* msgs) override {
      PYBIND11_OVERRIDE_PURE(
        double, /* Return type */
        stan::model::model_base,  /* Parent class */
        log_prob_propto,  /* Name of function in C++ */
        params_r, msgs
      );
    }

    // stan::math::var log_prob_propto(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) override {
    //   PYBIND11_OVERRIDE_PURE(
    //     stan::math::var, /* Return type */
    //     stan::model::model_base,  /* Parent class */
    //     log_prob_propto,  /* Name of function in C++ */
    //     params_r, msgs
    //   );
    // }

    double log_prob_propto_jacobian(Eigen::VectorXd& params_r, std::ostream* msgs) override {
      PYBIND11_OVERRIDE_PURE(
        double, /* Return type */
        stan::model::model_base,  /* Parent class */
        log_prob_propto_jacobian,  /* Name of function in C++ */
        params_r, msgs
      );
    }

    // stan::math::var log_prob_propto_jacobian(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) override {
    //   PYBIND11_OVERRIDE_PURE(
    //     stan::math::var, /* Return type */
    //     stan::model::model_base,  /* Parent class */
    //     log_prob_propto_jacobian,  /* Name of function in C++ */
    //     params_r, msgs
    //   );
    // }

    void transform_inits(const stan::io::var_context& context, Eigen::VectorXd& params_r, std::ostream* msgs) override {
      PYBIND11_OVERRIDE_PURE(
        void, /* Return type */
        stan::model::model_base,  /* Parent class */
        transform_inits,  /* Name of function in C++ */
        context, params_r, msgs
      );
    }

    void write_array(boost::ecuyer1988& base_rng,
                      Eigen::VectorXd& params_r,
                      Eigen::VectorXd& params_constrained_r,
                      bool include_tparams = true, bool include_gqs = true,
                      std::ostream* msgs = 0) override {
      PYBIND11_OVERRIDE_PURE(
        void, /* Return type */
        stan::model::model_base,  /* Parent class */
        write_array,  /* Name of function in C++ */
        base_rng, params_r, params_constrained_r, include_tparams, include_gqs, msgs
      );
    }
  };

  void model(py::module& m) {
  // stan::model::model_base
  py::class_<stan::model::model_base, py_model_base>(
    m, "model_base", "model_base class")
    .def(py::init<size_t>())
    .def("model_name", &stan::model::model_base::model_name, "Return the name of the model")
    .def("model_compile_info", &stan::model::model_base::model_compile_info, "Returns the compile information of the model")
    .def("get_param_names", &stan::model::model_base::get_param_names, "Return the name of the parameters")
    .def("get_dims", &stan::model::model_base::get_dims, "Return the dimensions of each parameter")
    .def("constrained_param_names", &stan::model::model_base::constrained_param_names, "Return constrained_param_names")
    .def("unconstrained_param_names", &stan::model::model_base::unconstrained_param_names, "Return unconstrained_param_names")
    .def("log_prob", &stan::model::model_base::log_prob, "Return the log density for the specified unconstrained parameters")
    .def("log_prob_jacobian", &stan::model::model_base::log_prob_jacobian, "Return the log density for the specified unconstrained parameters with Jacobian correction")
    .def("log_prob_propto", &stan::model::model_base::log_prob_propto, "Return the log density dropping normalizing constants")
    .def("log_prob_propto_jacobian", &stan::model::model_base::log_prob_propto_jacobian, "Return the log density dropping normalizing constants with Jacobian correction")
    .def("transform_inits", &stan::model::model_base::transform_inits, "Transform inits")
    .def("write_array", &stan::model::model_base::write_array, "Convert the specified sequence of unconstrained parameters to a sequence of constrained parameters");
}  // namespace stan_wrappers
