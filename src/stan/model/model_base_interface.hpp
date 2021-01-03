#ifndef STAN_MODEL_MODEL_BASE_INTERFACE_HPP
#define STAN_MODEL_MODEL_BASE_INTERFACE_HPP

#include <stan/model/model_base.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <stdexcept>

namespace stan {
namespace model {

class model_base_interface : public stan::model::model_base {
 public:
  /**
   * Construct a model with the specified number of real unconstrained
   * parameters.
   *
   * @param[in] num_params_r number of real unconstrained parameters
   */
  explicit model_base_interface(size_t num_params_r) : model_base(num_params_r) {}

  /**
   * Destructor.  This class has a no-op destructor.
   */
  virtual ~model_base_interface() {}

  std::vector<std::string> model_compile_info() const override {
    std::vector<std::string> stanc_info;
    stanc_info.push_back("custom model");
    return stanc_info;
  }

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian and with normalizing constants for
   * probability functions.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob(std::vector<double>& params_r,
                         std::ostream* msgs) const = 0;

  virtual double log_prob(Eigen::VectorXd& params_r,
                         std::ostream* msgs) const override = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, and its gradient.
   *
   * @param[in] params_r unconstrained parameters
   * @param[out] gradient Vector into which gradient is written.
   * @param[in,out] msgs message stream
   * @param[in] propto `true` if normalizing constants should be dropped
   * and result returned up to an additive constant
   * @param[in] jacobian_adjust_transform `true` if the log Jacobian adjustment is
   * included for the change of variables from unconstrained to
   * constrained parameters.
   * @return log density for specified parameters
   */
  virtual double log_prob_grad(std::vector<double>& params_r, std::vector<double>& gradient,
                         bool propto, bool jacobian_adjust_transform,
                         std::ostream* msgs) const = 0;

  virtual double log_prob_grad(Eigen::VectorXd& params_r, Eigen::VectorXd& gradient,
                         bool propto, bool jacobian_adjust_transform,
                         std::ostream* msgs) const = 0;

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and with
   * normalizing constants for probability functions.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_jacobian(std::vector<double>& params_r,
                                  std::ostream* msgs) const {
    return log_prob(params_r, msgs);
  }

  virtual double log_prob_jacobian(Eigen::VectorXd& params_r,
                                  std::ostream* msgs) const override {
    return log_prob(params_r, msgs);
  }

  /**
   * Return the log density for the specified unconstrained
   * parameters, without Jacobian correction for constraints and
   * dropping normalizing constants.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_propto(std::vector<double>& params_r,
                                std::ostream* msgs) const {
    return log_prob(params_r, msgs);
  }

  virtual double log_prob_propto(Eigen::VectorXd& params_r,
                                std::ostream* msgs)  const override {
    return log_prob(params_r, msgs);
  }

  /**
   * Return the log density for the specified unconstrained
   * parameters, with Jacobian correction for constraints and dropping
   * normalizing constants.
   *
   * <p>The Jacobian is of the inverse transform from unconstrained
   * parameters to constrained parameters; full details for Stan
   * language types can be found in the language reference manual.
   *
   * @param[in] params_r unconstrained parameters
   * @param[in,out] msgs message stream
   * @return log density for specified parameters
   */
  virtual double log_prob_propto_jacobian(std::vector<double>& params_r,
                                         std::ostream* msgs) const {
    return log_prob_jacobian(params_r, msgs);
  }

  virtual double log_prob_propto_jacobian(Eigen::VectorXd& params_r,
                                         std::ostream* msgs) const override {
    return log_prob_jacobian(params_r, msgs);
  }

  /**
   * Read constrained parameter values from the specified context,
   * unconstrain them, then concatenate the unconstrained sequences
   * into the specified parameter sequence.  Output messages go to the
   * specified stream.
   *
   * @param[in] context definitions of variable values
   * @param[in,out] params_r unconstrained parameter values produced
   * @param[in,out] msgs stream to which messages are written
   */
  virtual void transform_inits(const io::var_context& context,
                               std::vector<double>& params_r,
                               std::ostream* msgs) const = 0;

  virtual void transform_inits(const io::var_context& context,
                               Eigen::VectorXd& params_r,
                               std::ostream* msgs) const override = 0;

  /**
   * Convert the specified sequence of unconstrained parameters to a
   * sequence of constrained parameters, optionally including
   * transformed parameters and including generated quantities.  The
   * generated quantities may use the random number generator.  Any
   * messages are written to the specified stream.  The output
   * parameter sequence will be resized if necessary to match the
   * number of constrained scalar parameters.
   *
   * @param base_rng RNG to use for generated quantities
   * @param[in] params_r unconstrained parameters input
   * @param[in,out] params_r_constrained constrained parameters produced
   * @param[in] include_tparams true if transformed parameters are
   * included in output
   * @param[in] include_gqs true if generated quantities are included
   * in output
   * @param[in,out] msgs msgs stream to which messages are written
   */
  virtual void write_array(boost::ecuyer1988& rng, std::vector<double>& params_r,
                   std::vector<double>& params_r_constrained,
                   bool include_tparams = true, bool include_gqs = true,
                   std::ostream* msgs = 0) const = 0;

  virtual void write_array(boost::ecuyer1988& rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_r_constrained, bool include_tparams = true,
                   bool include_gqs = true,
                   std::ostream* msgs = 0) const override = 0;

  // Now non-virtual overrides

  double log_prob(std::vector<double>& params_r,
                  std::vector<int>& params_i,
                  std::ostream* msgs) const override {
    (void)params_i;  // unused
    return log_prob(params_r, msgs);
  }

  double log_prob_jacobian(std::vector<double>& params_r,
                                  std::vector<int>& params_i,
                                  std::ostream* msgs) const override {
    (void)params_i;  // unused
    return log_prob_jacobian(params_r, msgs);
  }

  double log_prob_propto(std::vector<double>& params_r,
                         std::vector<int>& params_i,
                         std::ostream* msgs) const override {
    (void)params_i;  // unused
    return log_prob_propto(params_r, msgs);
  }

  double log_prob_propto_jacobian(std::vector<double>& params_r,
                                  std::vector<int>& params_i,
                                  std::ostream* msgs) const override {
    (void)params_i;  // unused
    return log_prob_propto_jacobian(params_r, msgs);
  }

  virtual void transform_inits(const io::var_context& context,
                               std::vector<int>& params_i,
                               std::vector<double>& params_r,
                               std::ostream* msgs) const override {
    (void)params_i;  // unused
    return transform_inits(context, params_r, msgs);
  }

  void write_array(boost::ecuyer1988& rng, std::vector<double>& params_r,
                   std::vector<int>& params_i, std::vector<double>& params_r_constrained,
                   bool include_tparams = true, bool include_gqs = true,
                   std::ostream* msgs = 0) const override {
    (void)params_i;  // unused
    return write_array(rng, params_r, params_r_constrained, include_tparams, include_gqs, msgs);
  }

  // math::var returns are not supported here
  // in the future it could be possible to use precomputed_gradients or reverse_pass_callback

  inline math::var log_prob(std::vector<math::var>& theta,
                            std::vector<int>& theta_i,
                            std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob(std::vector<math::var>& theta, std::vector<int>& theta_i, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob_jacobian(std::vector<math::var>& theta,
                                     std::vector<int>& theta_i,
                                     std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob_jacobian(std::vector<math::var>& theta, std::vector<int>& theta_i, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob_propto(std::vector<math::var>& theta,
                                   std::vector<int>& theta_i,
                                   std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob_propto(std::vector<math::var>& theta, std::vector<int>& theta_i, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob_propto_jacobian(std::vector<math::var>& theta,
                                            std::vector<int>& theta_i,
                                            std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob_propto_jacobian(std::vector<math::var>& theta, std::vector<int>& theta_i, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob(Eigen::Matrix<math::var, -1, 1>& theta,
                            std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob(Eigen::Matrix<math::var, -1, 1>& theta, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob_jacobian(Eigen::Matrix<math::var, -1, 1>& theta,
                                     std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob_jacobian(Eigen::Matrix<math::var, -1, 1>& theta, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob_propto(Eigen::Matrix<math::var, -1, 1>& theta,
                                   std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob_propto(Eigen::Matrix<math::var, -1, 1>& theta, std::ostream* msgs) is not implemented!");
  }

  inline math::var log_prob_propto_jacobian(
      Eigen::Matrix<math::var, -1, 1>& theta,
      std::ostream* msgs) const override {
    throw std::runtime_error("math::var log_prob_propto_jacobian(Eigen::Matrix<math::var, -1, 1>& theta, std::ostream* msgs) is not implemented!");
  }
};

}  // namespace model
}  // namespace stan
#endif  // STAN_MODEL_MODEL_BASE_INTERFACE_HPP
