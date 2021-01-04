#include <stan/model/log_prob_propto.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/model/test_model_interface_rosenbrock.hpp>

TEST(ModelUtil, streams) {
  stan::test::capture_std_streams();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  stan_model model(data_var_context, 0, static_cast<std::stringstream*>(0));
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;

  std::stringstream out;

  try {
    stan::model::log_prob_propto<true, stan_model>(model, params_r, params_i,
                                                   0);
    stan::model::log_prob_propto<false, stan_model>(model, params_r, params_i,
                                                    0);
    out.str("");
    stan::model::log_prob_propto<true, stan_model>(model, params_r, params_i,
                                                   &out);
    stan::model::log_prob_propto<false, stan_model>(model, params_r, params_i,
                                                    &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_propto";
  }

  try {
    Eigen::VectorXd p(1);
    stan::model::log_prob_propto<true, stan_model>(model, p, 0);
    stan::model::log_prob_propto<false, stan_model>(model, p, 0);
    out.str("");
    stan::model::log_prob_propto<true, stan_model>(model, p, &out);
    stan::model::log_prob_propto<false, stan_model>(model, p, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_propto";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}

TEST(ModelUtil, logProbProptoModelBaseInterface) {
  stan::test::capture_std_streams();

  int dim = 2;

  rosenbrock_model model(dim);

  std::vector<double> params_r(dim);
  params_r[0] = 0.5;
  params_r[1] = 0.5;
  std::vector<int> params_i(0);
  double result;

  std::stringstream out;

  try {
    result = stan::model::log_prob_propto<true, rosenbrock_model>(
        model, params_r, params_i, 0);
    EXPECT_FLOAT_EQ(-6.5, result);
    result = stan::model::log_prob_propto<false, rosenbrock_model>(
        model, params_r, params_i, 0);
    EXPECT_FLOAT_EQ(-6.5, result);
    out.str("");
    result = stan::model::log_prob_propto<true, rosenbrock_model>(
        model, params_r, params_i, &out);
    EXPECT_FLOAT_EQ(-6.5, result);
    result = stan::model::log_prob_propto<false, rosenbrock_model>(
        model, params_r, params_i, &out);
    EXPECT_FLOAT_EQ(-6.5, result);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_propto";
  }

  try {
    Eigen::VectorXd p(dim);
    p << 0.5, 0.5;
    result = stan::model::log_prob_propto<true, rosenbrock_model>(model, p, 0);
    EXPECT_FLOAT_EQ(-6.5, result);
    result = stan::model::log_prob_propto<false, rosenbrock_model>(model, p, 0);
    EXPECT_FLOAT_EQ(-6.5, result);
    out.str("");
    result
        = stan::model::log_prob_propto<true, rosenbrock_model>(model, p, &out);
    EXPECT_FLOAT_EQ(-6.5, result);
    result
        = stan::model::log_prob_propto<false, rosenbrock_model>(model, p, &out);
    EXPECT_FLOAT_EQ(-6.5, result);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_propto";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
