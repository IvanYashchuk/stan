#ifndef STAN_MODEL_GRADIENT_HPP
#define STAN_MODEL_GRADIENT_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/model_functional.hpp>
#include <stan/model/model_base_interface.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace stan {
namespace model {
namespace internal {

template <class M>
void gradient_impl(const M& model,
                   const Eigen::Matrix<double, Eigen::Dynamic, 1>& x, double& f,
                   Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                   std::ostream* msgs = 0) {
  stan::math::gradient(model_functional<M>(model, msgs), x, f, grad_f);
}

template <class M>
void gradient_impl(const M& model,
                   const Eigen::Matrix<double, Eigen::Dynamic, 1>& x, double& f,
                   Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                   callbacks::logger& logger) {
  std::stringstream ss;
  try {
    stan::math::gradient(model_functional<M>(model, &ss), x, f, grad_f);
  } catch (std::exception& e) {
    if (ss.str().length() > 0)
      logger.info(ss);
    throw;
  }
  if (ss.str().length() > 0)
    logger.info(ss);
}

template <class M, typename Enable = void>
struct GradientHelper {
  static void gradient(const M& model,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                       double& f,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                       std::ostream* msgs = 0) {
    return gradient_impl<M>(model, x, f, grad_f, msgs);
  }

  static void gradient(const M& model,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                       double& f,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                       callbacks::logger& logger) {
    return gradient_impl<M>(model, x, f, grad_f, logger);
  }
};

template <class M>
struct GradientHelper<M, enable_if_derived_interface_t<M>> {
  static void gradient(const stan::model::model_base_interface& model,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                       double& f,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                       std::ostream* msgs = 0) {
    f = stan::model::log_prob_grad<true, true>(
        model, const_cast<Eigen::VectorXd&>(x), grad_f, msgs);
  }

  static void gradient(const stan::model::model_base_interface& model,
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                       double& f,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                       callbacks::logger& logger) {
    std::stringstream ss;
    try {
      f = stan::model::log_prob_grad<true, true>(
          model, const_cast<Eigen::VectorXd&>(x), grad_f, &ss);
    } catch (std::exception& e) {
      if (ss.str().length() > 0) {
        logger.info(ss);
      }
      throw;
    }
    if (ss.str().length() > 0) {
      logger.info(ss);
    }
  }
};
}  // namespace internal

template <class M>
void gradient(const M& model, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
              double& f, Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
              std::ostream* msgs = 0) {
  return internal::GradientHelper<M>::gradient(model, x, f, grad_f, msgs);
}

template <class M>
void gradient(const M& model, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
              double& f, Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
              callbacks::logger& logger) {
  return internal::GradientHelper<M>::gradient(model, x, f, grad_f, logger);
}

}  // namespace model
}  // namespace stan
#endif
