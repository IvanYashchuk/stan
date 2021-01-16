#ifndef STAN_MODEL_LOG_PROB_HPP
#define STAN_MODEL_LOG_PROB_HPP

#include <stan/model/model_base_interface.hpp>
#include <iostream>
#include <vector>

namespace stan {
namespace model {
namespace internal {

template <bool propto, bool jacobian_adjust_transform, class M>
double log_prob_impl(const M& model, std::vector<double>& params_r,
                     std::vector<int>& params_i, std::ostream* msgs = 0) {
  return model.template log_prob<propto, jacobian_adjust_transform>(
      params_r, params_i, msgs);
}

// Here we want to call specific overloads if model is a derived class of
// stan::model::model_base_interface partial template specialization of
// functions is not possible in C++, therefore we create a helper struct

// This is the general template wrapper of log_prob_propto_impl
template <bool propto, bool jacobian_adjust_transform, class M,
          typename Enable = void>
struct LogProbHelper {
  static double log_prob(const M& model, std::vector<double>& params_r,
                         std::vector<int>& params_i, std::ostream* msgs = 0) {
    return log_prob_impl<propto, jacobian_adjust_transform, M>(model, params_r,
                                                               params_i, msgs);
  }
};

// This is the partial template specialization for derived classes of
// stan::model::model_base_interface
template <bool propto, bool jacobian_adjust_transform, class M>
struct LogProbHelper<propto, jacobian_adjust_transform, M,
                     enable_if_derived_interface_t<M>> {
  static double log_prob(const stan::model::model_base_interface& model,
                         std::vector<double>& params_r,
                         std::vector<int>& params_i, std::ostream* msgs = 0) {
    if (propto && jacobian_adjust_transform)
      return model.log_prob_propto_jacobian(params_r, params_i, msgs);
    else if (propto && !jacobian_adjust_transform)
      return model.log_prob_propto(params_r, params_i, msgs);
    else if (!propto && jacobian_adjust_transform)
      return model.log_prob_jacobian(params_r, params_i, msgs);
    else  // if (!propto && !jacobian_adjust_transform)
      return model.log_prob(params_r, params_i, msgs);
  }
};

}  // namespace internal

/**
 * Helper function to calculate log probability for
 * <code>double</code> scalars.
 *
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
template <bool propto, bool jacobian_adjust_transform, class M>
double log_prob(const M& model, std::vector<double>& params_r,
                std::vector<int>& params_i, std::ostream* msgs = 0) {
  return internal::LogProbHelper<propto, jacobian_adjust_transform,
                                 M>::log_prob(model, params_r, params_i, msgs);
}

}  // namespace model
}  // namespace stan
#endif
