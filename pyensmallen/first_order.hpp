#pragma once

#include <ensmallen_bits/adam/adam.hpp>
#include "utils.hpp"

// Template class for Adam-type optimizers
template <typename UpdateRule>
class PyAdamType
{
public:
  PyAdamType() : optimizer() {}
  PyAdamType(double stepSize, size_t batchSize)
      : optimizer(stepSize, batchSize) {}
  PyAdamType(double stepSize, size_t batchSize, double beta1, double beta2,
             double eps, size_t maxIterations, double tolerance, bool shuffle,
             bool resetPolicy, bool exactObjective)
      : optimizer(stepSize, batchSize, beta1, beta2, eps, maxIterations,
                  tolerance, shuffle, resetPolicy, exactObjective) {}
  double getStepSize() const { return optimizer.StepSize(); }
  void setStepSize(double stepSize) { optimizer.StepSize() = stepSize; }

  size_t getBatchSize() const { return optimizer.BatchSize(); }
  void setBatchSize(size_t batchSize) { optimizer.BatchSize() = batchSize; }

  double getBeta1() const { return optimizer.Beta1(); }
  void setBeta1(double beta1) { optimizer.Beta1() = beta1; }

  double getBeta2() const { return optimizer.Beta2(); }
  void setBeta2(double beta2) { optimizer.Beta2() = beta2; }

  double getEpsilon() const { return optimizer.Epsilon(); }
  void setEpsilon(double eps) { optimizer.Epsilon() = eps; }

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }
  void setMaxIterations(size_t maxIterations)
  {
    optimizer.MaxIterations() = maxIterations;
  }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  bool getShuffle() const { return optimizer.Shuffle(); }
  void setShuffle(bool shuffle) { optimizer.Shuffle() = shuffle; }

  bool getExactObjective() const { return optimizer.ExactObjective(); }
  void setExactObjective(bool exactObjective)
  {
    optimizer.ExactObjective() = exactObjective;
  }

  bool getResetPolicy() const { return optimizer.ResetPolicy(); }
  void setResetPolicy(bool resetPolicy)
  {
    optimizer.ResetPolicy() = resetPolicy;
  }

  py::array_t<double> Optimize(DifferentiableFunction f,
                               py::array_t<double> initial_point)
  {
    py::buffer_info buf_info = initial_point.request();
    arma::vec arma_initial_point(static_cast<double *>(buf_info.ptr),
                                 buf_info.shape[0], false, true);

    DifferentiableFunctionWrapper fw(f);
    arma::vec result = arma_initial_point;

    optimizer.Optimize(fw, result);

    return py::array_t<double>(result.n_elem, result.memptr());
  }

private:
  ens::AdamType<UpdateRule> optimizer;
};