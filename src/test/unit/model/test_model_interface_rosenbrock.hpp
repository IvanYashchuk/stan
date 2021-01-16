#ifndef TEST_MODEL_INERFACE_ROSENBROCK_TEST_MODEL_HPP
#define TEST_MODEL_INERFACE_ROSENBROCK_TEST_MODEL_HPP

#include <stan/math/prim.hpp>
#include <stan/model/model_base_interface.hpp>
#include <ostream>
#include <string>
#include <vector>

class rosenbrock_model : public stan::model::model_base_interface {
 public:
  rosenbrock_model(size_t n) : model_base_interface(n) {}

  virtual ~rosenbrock_model() {}

  using model_base_interface::transform_inits;
  using model_base_interface::write_array;

  std::string model_name() const override { return "rosenbrock_model"; }

  void get_param_names(std::vector<std::string>& names) const override {
    names.clear();
    names.emplace_back("xy");
  }

  void get_dims(std::vector<std::vector<size_t> >& dimss) const override {
    dimss.clear();
    dimss.emplace_back(std::vector<size_t>{static_cast<size_t>(2)});
  }

  void constrained_param_names(std::vector<std::string>& param_names,
                               bool include_tparams,
                               bool include_gqs) const override {
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      param_names.emplace_back(std::string() + "xy" + '.'
                               + std::to_string(sym1__));
    }
  }

  void unconstrained_param_names(std::vector<std::string>& param_names,
                                 bool include_tparams,
                                 bool include_gqs) const override {
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      param_names.emplace_back(std::string() + "xy" + '.'
                               + std::to_string(sym1__));
    }
  }

  double log_prob(std::vector<double>& params_r,
                  std::ostream* msgs) const override {
    double* x = params_r.data();
    int nn = 1;
    double f = 0;
    double alpha = 100.0;

    // Compute 2D Rosenbrock function
    for (int i = 0; i < nn; i++) {
      double t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      double t2 = 1 - x[2 * i];
      f += alpha * t1 * t1 + t2 * t2;
    }
    return -f;
  }

  double log_prob(Eigen::VectorXd& params_r,
                  std::ostream* msgs) const override {
    double* x = params_r.data();
    int nn = 1;
    double f = 0;
    double alpha = 100.0;

    // Compute 2D Rosenbrock function
    for (int i = 0; i < nn; i++) {
      double t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      double t2 = 1 - x[2 * i];
      f += alpha * t1 * t1 + t2 * t2;
    }
    return -f;
  }

  double log_prob_grad(std::vector<double>& params_r,
                       std::vector<double>& gradient, bool propto,
                       bool jacobian_adjust_transform,
                       std::ostream* msgs) const override {
    gradient.clear();
    gradient.resize(num_params_r());

    double* x = params_r.data();
    double* g = gradient.data();
    int nn = 1;
    double f = 0;
    double alpha = 100.0;

    // Compute 2D Rosenbrock function and its gradient
    for (int i = 0; i < nn; i++) {
      double t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      double t2 = 1 - x[2 * i];
      f += alpha * t1 * t1 + t2 * t2;
      g[2 * i] = -(-4 * alpha * t1 * x[2 * i] - 2.0 * t2);
      g[2 * i + 1] = -(2 * alpha * t1);
    }
    return -f;
  }

  double log_prob_grad(Eigen::VectorXd& params_r, Eigen::VectorXd& gradient,
                       bool propto, bool jacobian_adjust_transform,
                       std::ostream* msgs) const override {
    gradient.resize(num_params_r());

    double* x = params_r.data();
    double* g = gradient.data();
    int nn = 1;
    double f = 0;
    double alpha = 100.0;

    // Compute 2D Rosenbrock function and its gradient
    for (int i = 0; i < nn; i++) {
      double t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      double t2 = 1 - x[2 * i];
      f += alpha * t1 * t1 + t2 * t2;
      g[2 * i] = -(-4 * alpha * t1 * x[2 * i] - 2.0 * t2);
      g[2 * i + 1] = -(2 * alpha * t1);
    }
    return -f;
  }

  void transform_inits(const stan::io::var_context& context,
                       std::vector<double>& params_r,
                       std::ostream* msgs) const override {
    params_r.clear();
    params_r.resize(num_params_r());

    std::vector<double> xy_vector = context.vals_r("xy");
    for (int i = 0; i < num_params_r(); i++) {
      params_r[i] = xy_vector[i];
    }
  }

  void transform_inits(const stan::io::var_context& context,
                       Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {
    std::vector<double> xy_vector = context.vals_r("xy");
    for (int i = 0; i < num_params_r(); i++) {
      params_r[i] = xy_vector[i];
    }
  }

  void write_array(boost::ecuyer1988& base_rng, std::vector<double>& params_r,
                   std::vector<double>& params_constrained_r,
                   bool include_tparams, bool include_gqs,
                   std::ostream* msgs) const override {
    params_constrained_r.clear();
    params_constrained_r.resize(num_params_r());
    for (int i = 0; i < num_params_r(); i++) {
      params_constrained_r[i] = params_r[i];
    }
  }

  void write_array(boost::ecuyer1988& base_rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_constrained_r, bool include_tparams,
                   bool include_gqs, std::ostream* msgs) const override {
    params_constrained_r = params_r;
  }
};

#endif
