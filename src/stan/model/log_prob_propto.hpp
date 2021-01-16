#ifndef STAN_MODEL_LOG_PROB_PROPTO_HPP
#define STAN_MODEL_LOG_PROB_PROPTO_HPP

#include <stan/math/rev.hpp>
#include <stan/model/model_base_interface.hpp>
#include <iostream>
#include <vector>

namespace stan {
namespace model {
namespace internal {

/**
 * Helper function to calculate log probability for
 * <code>double</code> scalars up to a proportion.
 *
 * This implementation wraps the <code>double</code> values in
 * <code>stan::math::var</code> and calls the model's
 * <code>log_prob()</code> function with <code>propto=true</code>
 * and the specified parameter for applying the Jacobian
 * adjustment for transformed parameters.
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
 * @param[in,out] msgs
 */
template <bool jacobian_adjust_transform, class M>
double log_prob_propto_impl(const M& model, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::ostream* msgs = 0) {
  using stan::math::var;
  using std::vector;
  try {
    vector<var> ad_params_r;
    ad_params_r.reserve(model.num_params_r());
    for (size_t i = 0; i < model.num_params_r(); ++i)
      ad_params_r.push_back(params_r[i]);
    double lp = model
                    .template log_prob<true, jacobian_adjust_transform>(
                        ad_params_r, params_i, msgs)
                    .val();
    stan::math::recover_memory();
    return lp;
  } catch (std::exception& ex) {
    stan::math::recover_memory();
    throw;
  }
}

/**
 * Helper function to calculate log probability for
 * <code>double</code> scalars up to a proportion.
 *
 * This implementation wraps the <code>double</code> values in
 * <code>stan::math::var</code> and calls the model's
 * <code>log_prob()</code> function with <code>propto=true</code>
 * and the specified parameter for applying the Jacobian
 * adjustment for transformed parameters.
 *
 * @tparam propto True if calculation is up to proportion
 * (double-only terms dropped).
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[in,out] msgs
 */
template <bool jacobian_adjust_transform, class M>
double log_prob_propto_impl(const M& model, Eigen::VectorXd& params_r,
                            std::ostream* msgs = 0) {
  using stan::math::var;
  using std::vector;
  vector<int> params_i(0);
  try {
    vector<var> ad_params_r;
    ad_params_r.reserve(model.num_params_r());
    for (size_t i = 0; i < model.num_params_r(); ++i)
      ad_params_r.push_back(params_r(i));
    double lp = model
                    .template log_prob<true, jacobian_adjust_transform>(
                        ad_params_r, params_i, msgs)
                    .val();
    stan::math::recover_memory();
    return lp;
  } catch (std::exception& ex) {
    stan::math::recover_memory();
    throw;
  }
}

// Here we want to call specific overloads if model is a derived class of
// stan::model::model_base_interface partial template specialization of
// functions is not possible in C++, therefore we create a helper struct

// This is the general template wrapper of log_prob_propto_impl
template <bool jacobian_adjust_transform, class M, typename Enable = void>
struct LogProbHelper {
  static double log_prob_propto(const M& model, std::vector<double>& params_r,
                                std::vector<int>& params_i,
                                std::ostream* msgs = 0) {
    return log_prob_propto_impl<jacobian_adjust_transform, M>(model, params_r,
                                                              params_i, msgs);
  }

  static double log_prob_propto(const M& model, Eigen::VectorXd& params_r,
                                std::ostream* msgs = 0) {
    return log_prob_propto_impl<jacobian_adjust_transform, M>(model, params_r,
                                                              msgs);
  }
};

// This is the partial template specialization for derived classes of
// stan::model::model_base_interface
template <bool jacobian_adjust_transform, class M>
struct LogProbHelper<jacobian_adjust_transform, M,
                     enable_if_derived_interface_t<M>> {
  static double log_prob_propto(const stan::model::model_base_interface& model,
                                std::vector<double>& params_r,
                                std::vector<int>& params_i,
                                std::ostream* msgs = 0) {
    if (jacobian_adjust_transform) {
      return model.log_prob_propto_jacobian(params_r, params_i, msgs);
    }  // else if !jacobian_adjust_transform
    return model.log_prob_propto(params_r, params_i, msgs);
  }

  static double log_prob_propto(const stan::model::model_base_interface& model,
                                Eigen::VectorXd& params_r,
                                std::ostream* msgs = 0) {
    if (jacobian_adjust_transform) {
      return model.log_prob_propto_jacobian(params_r, msgs);
    }  // else if !jacobian_adjust_transform
    return model.log_prob_propto(params_r, msgs);
  }
};

}  // namespace internal

/**
 * Helper function to calculate log probability for
 * <code>double</code> scalars up to a proportion.
 *
 * This implementation calls the model's
 * <code>log_prob()</code> function with <code>propto=true</code>
 * and the specified parameter for applying the Jacobian
 * adjustment for transformed parameters.
 *
 * @tparam propto True if calculation is up to proportion
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[in] params_i Integer-valued parameters.
 * @param[in,out] msgs
 */
template <bool jacobian_adjust_transform, class M>
double log_prob_propto(const M& model, std::vector<double>& params_r,
                       std::vector<int>& params_i, std::ostream* msgs = 0) {
  return internal::LogProbHelper<jacobian_adjust_transform, M>::log_prob_propto(
      model, params_r, params_i, msgs);
}

/**
 * Helper function to calculate log probability for
 * <code>double</code> scalars up to a proportion.
 *
 * This implementation calls the model's
 * <code>log_prob()</code> function with <code>propto=true</code>
 * and the specified parameter for applying the Jacobian
 * adjustment for transformed parameters.
 *
 * @tparam propto True if calculation is up to proportion
 * @tparam jacobian_adjust_transform True if the log absolute
 * Jacobian determinant of inverse parameter transforms is added to
 * the log probability.
 * @tparam M Class of model.
 * @param[in] model Model.
 * @param[in] params_r Real-valued parameters.
 * @param[in,out] msgs
 */
template <bool jacobian_adjust_transform, class M>
double log_prob_propto(const M& model, Eigen::VectorXd& params_r,
                       std::ostream* msgs = 0) {
  return internal::LogProbHelper<jacobian_adjust_transform, M>::log_prob_propto(
      model, params_r, msgs);
}

}  // namespace model
}  // namespace stan
#endif
