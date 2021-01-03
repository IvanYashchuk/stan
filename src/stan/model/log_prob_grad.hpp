#ifndef STAN_MODEL_LOG_PROB_GRAD_HPP
#define STAN_MODEL_LOG_PROB_GRAD_HPP

#include <stan/math/rev.hpp>
#include <stan/model/model_base_interface.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

namespace stan {
namespace model {
namespace internal {

/**
 * Compute the gradient using reverse-mode automatic
 * differentiation, writing the result into the specified
 * gradient, using the specified perturbation.
 *
 * @tparam propto True if calculation is up to proportion
 * (double-only terms dropped).
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[in] params_i Integer-valued parameters.
 * @param[out] gradient Vector into which gradient is written.
 * @param[in,out] msgs
 */
template <bool propto, bool jacobian_adjust_transform, class M>
double log_prob_grad_impl(const M& model, std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          std::vector<double>& gradient,
                          std::ostream* msgs = 0) {
  using stan::math::var;
  using std::vector;
  try {
    vector<var> ad_params_r(params_r.size());
    for (size_t i = 0; i < model.num_params_r(); ++i) {
      stan::math::var var_i(params_r[i]);
      ad_params_r[i] = var_i;
    }
    var adLogProb = model.template log_prob<propto, jacobian_adjust_transform>(
        ad_params_r, params_i, msgs);
    double lp = adLogProb.val();
    adLogProb.grad(ad_params_r, gradient);
    stan::math::recover_memory();
    return lp;
  } catch (const std::exception& ex) {
    stan::math::recover_memory();
    throw;
  }
}

/**
 * Compute the gradient using reverse-mode automatic
 * differentiation, writing the result into the specified
 * gradient, using the specified perturbation.
 *
 * @tparam propto True if calculation is up to proportion
 * (double-only terms dropped).
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[out] gradient Vector into which gradient is written.
 * @param[in,out] msgs
 */
template <bool propto, bool jacobian_adjust_transform, class M>
double log_prob_grad_impl(const M& model, Eigen::VectorXd& params_r,
                          Eigen::VectorXd& gradient, std::ostream* msgs = 0) {
  using stan::math::var;
  using std::vector;
  try {
    Eigen::Matrix<var, Eigen::Dynamic, 1> ad_params_r(params_r.size());
    for (size_t i = 0; i < model.num_params_r(); ++i) {
      stan::math::var var_i(params_r[i]);
      ad_params_r[i] = var_i;
    }
    var adLogProb = model.template log_prob<propto, jacobian_adjust_transform>(
        ad_params_r, msgs);
    double val = adLogProb.val();
    stan::math::grad(adLogProb, ad_params_r, gradient);
    stan::math::recover_memory();
    return val;
  } catch (std::exception& ex) {
    stan::math::recover_memory();
    throw;
  }
}

// Here we want to call specific overloads if model is a derived class of
// stan::model::model_base_interface partial template specialization of
// functions is not possible in C++, therefore we create a helper struct

template <bool propto, bool jacobian_adjust_transform>
double log_prob_grad_interface_impl(
    const stan::model::model_base_interface& model,
    std::vector<double>& params_r, std::vector<int>& params_i,
    std::vector<double>& gradient, std::ostream* msgs = 0) {
  (void)params_i;  // unused
  return model.log_prob_grad(params_r, gradient, propto,
                             jacobian_adjust_transform, msgs);
}

template <bool propto, bool jacobian_adjust_transform>
double log_prob_grad_interface_impl(
    const stan::model::model_base_interface& model, Eigen::VectorXd& params_r,
    Eigen::VectorXd& gradient, std::ostream* msgs = 0) {
  return model.log_prob_grad(params_r, gradient, propto,
                             jacobian_adjust_transform, msgs);
}

template <bool propto, bool jacobian_adjust_transform, class M,
          typename Enable = void>
struct Helper {
  static double log_prob_grad(const M& model, std::vector<double>& params_r,
                              std::vector<int>& params_i,
                              std::vector<double>& gradient,
                              std::ostream* msgs = 0) {
    return log_prob_grad_impl<propto, jacobian_adjust_transform, M>(
        model, params_r, params_i, gradient, msgs);
  }

  static double log_prob_grad(const M& model, Eigen::VectorXd& params_r,
                              Eigen::VectorXd& gradient,
                              std::ostream* msgs = 0) {
    return log_prob_grad_impl<propto, jacobian_adjust_transform, M>(
        model, params_r, gradient, msgs);
  }
};

template <bool propto, bool jacobian_adjust_transform, class M>
struct Helper<propto, jacobian_adjust_transform, M,
              typename std::enable_if<std::is_base_of<
                  stan::model::model_base_interface, M>::value>::type> {
  static double log_prob_grad(const stan::model::model_base_interface& model,
                              std::vector<double>& params_r,
                              std::vector<int>& params_i,
                              std::vector<double>& gradient,
                              std::ostream* msgs = 0) {
    return log_prob_grad_interface_impl<propto, jacobian_adjust_transform>(
        model, params_r, params_i, gradient, msgs);
  }

  static double log_prob_grad(const stan::model::model_base_interface& model,
                              Eigen::VectorXd& params_r,
                              Eigen::VectorXd& gradient,
                              std::ostream* msgs = 0) {
    return log_prob_grad_interface_impl<propto, jacobian_adjust_transform>(
        model, params_r, gradient, msgs);
  }
};
}  // namespace internal

/**
 * Compute the gradient using reverse-mode automatic
 * differentiation, writing the result into the specified
 * gradient, using the specified perturbation.
 *
 * @tparam propto True if calculation is up to proportion
 * (double-only terms dropped).
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[in] params_i Integer-valued parameters.
 * @param[out] gradient Vector into which gradient is written.
 * @param[in,out] msgs
 */
template <bool propto, bool jacobian_adjust_transform, class M>
double log_prob_grad(const M& model, std::vector<double>& params_r,
                     std::vector<int>& params_i, std::vector<double>& gradient,
                     std::ostream* msgs = 0) {
  return internal::Helper<propto, jacobian_adjust_transform, M>::log_prob_grad(
      model, params_r, params_i, gradient, msgs);
}

/**
 * Compute the gradient using reverse-mode automatic
 * differentiation, writing the result into the specified
 * gradient, using the specified perturbation.
 *
 * @tparam propto True if calculation is up to proportion
 * (double-only terms dropped).
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[out] gradient Vector into which gradient is written.
 * @param[in,out] msgs
 */
template <bool propto, bool jacobian_adjust_transform, class M>
double log_prob_grad(const M& model, Eigen::VectorXd& params_r,
                     Eigen::VectorXd& gradient, std::ostream* msgs = 0) {
  return internal::Helper<propto, jacobian_adjust_transform, M>::log_prob_grad(
      model, params_r, gradient, msgs);
}

}  // namespace model
}  // namespace stan
#endif
