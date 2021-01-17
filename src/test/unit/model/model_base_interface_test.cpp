#include <gtest/gtest.h>
#include <stan/model/model_base_interface.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct mock_model : public stan::model::model_base_interface {
  mock_model(size_t n) : model_base_interface(n) {}

  virtual ~mock_model() {}

  std::string model_name() const override { return "mock_model"; }

  void get_param_names(std::vector<std::string>& names) const override {}
  void get_dims(std::vector<std::vector<size_t> >& dimss) const override {}

  void constrained_param_names(std::vector<std::string>& param_names,
                               bool include_tparams,
                               bool include_gqs) const override {}

  void unconstrained_param_names(std::vector<std::string>& param_names,
                                 bool include_tparams,
                                 bool include_gqs) const override {}

  double log_prob(std::vector<double>& params_r,
                  std::ostream* msgs) const override {
    return 1;
  }

  double log_prob(Eigen::VectorXd& params_r,
                  std::ostream* msgs) const override {
    return 10;
  }

  double log_prob_jacobian(std::vector<double>& params_r,
                           std::ostream* msgs) const override {
    return 2;
  }

  double log_prob_jacobian(Eigen::VectorXd& params_r,
                           std::ostream* msgs) const override {
    return 20;
  }

  double log_prob_propto(std::vector<double>& params_r,
                         std::ostream* msgs) const override {
    return 3;
  }

  double log_prob_propto(Eigen::VectorXd& params_r,
                         std::ostream* msgs) const override {
    return 30;
  }

  double log_prob_propto_jacobian(std::vector<double>& params_r,
                                  std::ostream* msgs) const override {
    return 4;
  }

  double log_prob_propto_jacobian(Eigen::VectorXd& params_r,
                                  std::ostream* msgs) const override {
    return 40;
  }

  void convert_to_unconstrained(const stan::io::var_context& context,
                                std::vector<double>& params_r,
                                std::ostream* msgs) const override {}

  void convert_to_unconstrained(const stan::io::var_context& context,
                                Eigen::VectorXd& params_r,
                                std::ostream* msgs) const override {}

  void convert_to_constrained(boost::ecuyer1988& base_rng,
                              const std::vector<double>& params_r,
                              std::vector<double>& params_constrained_r,
                              bool include_tparams, bool include_gqs,
                              std::ostream* msgs) const override {}

  void convert_to_constrained(boost::ecuyer1988& base_rng,
                              const Eigen::VectorXd& params_r,
                              Eigen::VectorXd& params_constrained_r,
                              bool include_tparams, bool include_gqs,
                              std::ostream* msgs) const override {}

  double log_prob_grad(std::vector<double>& params_r,
                       std::vector<double>& gradient, bool propto,
                       bool jacobian_adjust_transform,
                       std::ostream* msgs) const override {
    return 0;
  }

  double log_prob_grad(Eigen::VectorXd& params_r, Eigen::VectorXd& gradient,
                       bool propto, bool jacobian_adjust_transform,
                       std::ostream* msgs) const override {
    return 0;
  }
};

TEST(model, modelBaseInterfaceInheritance) {
  // check that prob_grad inheritance works
  mock_model m(17);
  EXPECT_EQ(17u, m.num_params_r());
  EXPECT_EQ(0u, m.num_params_i());
  EXPECT_THROW(m.param_range_i(0), std::out_of_range);
}

TEST(model, modelInterfaceTemplateLogProb) {
  mock_model m(17);
  stan::model::model_base& bm = m;
  Eigen::VectorXd params_r_eigen(2);
  std::vector<double> params_r_vector(2);
  std::vector<int> params_i_vector;
  std::stringstream ss;
  std::ostream* msgs = &ss;

  // these versions defined in mock_model; make sure they work from base
  EXPECT_FLOAT_EQ(10, bm.log_prob(params_r_eigen, msgs));
  EXPECT_FLOAT_EQ(1, bm.log_prob(params_r_vector, params_i_vector, msgs));
  EXPECT_FLOAT_EQ(20, bm.log_prob_jacobian(params_r_eigen, msgs));
  EXPECT_FLOAT_EQ(2,
                  bm.log_prob_jacobian(params_r_vector, params_i_vector, msgs));
  EXPECT_FLOAT_EQ(30, bm.log_prob_propto(params_r_eigen, msgs));
  EXPECT_FLOAT_EQ(3,
                  bm.log_prob_propto(params_r_vector, params_i_vector, msgs));
  EXPECT_FLOAT_EQ(40, bm.log_prob_propto_jacobian(params_r_eigen, msgs));
  EXPECT_FLOAT_EQ(
      4, bm.log_prob_propto_jacobian(params_r_vector, params_i_vector, msgs));

  // test template version from base class;  not callable from mock_model
  // because templated class functions are not inherited
  // long form assignment avoids test macro parse error with multi tparams
  double v1 = bm.template log_prob<false, false>(params_r_eigen, msgs);
  EXPECT_FLOAT_EQ(10, v1);
  double v2 = bm.template log_prob<false, true>(params_r_eigen, msgs);
  EXPECT_FLOAT_EQ(20, v2);
  double v3 = bm.template log_prob<true, false>(params_r_eigen, msgs);
  EXPECT_FLOAT_EQ(30, v3);
  double v4 = bm.template log_prob<true, true>(params_r_eigen, msgs);
  EXPECT_FLOAT_EQ(40, v4);

  double v5 = bm.template log_prob<false, false>(params_r_vector,
                                                 params_i_vector, msgs);
  EXPECT_FLOAT_EQ(1, v5);
  double v6 = bm.template log_prob<false, true>(params_r_vector,
                                                params_i_vector, msgs);
  EXPECT_FLOAT_EQ(2, v6);
  double v7 = bm.template log_prob<true, false>(params_r_vector,
                                                params_i_vector, msgs);
  EXPECT_FLOAT_EQ(3, v7);
  double v8 = bm.template log_prob<true, true>(params_r_vector, params_i_vector,
                                               msgs);
  EXPECT_FLOAT_EQ(4, v8);
}
