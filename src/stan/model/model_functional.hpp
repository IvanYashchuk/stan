#ifndef STAN_MODEL_MODEL_FUNCTIONAL_HPP
#define STAN_MODEL_MODEL_FUNCTIONAL_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/model/model_base_interface.hpp>
#include <iostream>

namespace stan {
namespace model {

// Interface for automatic differentiation of models
template <class M, typename Enable = void>
struct model_functional {
  const M& model;
  std::ostream* o;

  model_functional(const M& m, std::ostream* out) : model(m), o(out) {}

  template <typename T>
  T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    // log_prob() requires non-const but doesn't modify its argument
    return model.template log_prob<true, true, T>(
        const_cast<Eigen::Matrix<T, -1, 1>&>(x), o);
  }
};

// model_functional is not supported to work with derived class of
// model_base_interface
template <class M>
struct model_functional<M, enable_if_derived_interface_t<M>> {
  const M& model;
  std::ostream* o;

  model_functional(const M& m, std::ostream* out) : model(m), o(out) {
    std::ostringstream error_msg;
    error_msg << "model_functional is not supported to work with '"
              << typeid(model).name() << "'";
    throw std::runtime_error(error_msg.str());
  }
};

}  // namespace model
}  // namespace stan
#endif
