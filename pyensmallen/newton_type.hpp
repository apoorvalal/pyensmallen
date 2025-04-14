#pragma once

#include "utils.hpp"
#include "report.hpp"

// Wrapper for L-BFGS optimizer
class PyL_BFGS
{
public:
  PyL_BFGS() : optimizer() {}
  PyL_BFGS(size_t numBasis, size_t maxIterations)
  {
    optimizer = ens::L_BFGS(numBasis, maxIterations);
  }
  PyL_BFGS(size_t numBasis, size_t maxIterations, double armijoConstant,
           double wolfe, double minGradientNorm, double factr,
           size_t maxLineSearchTrials)
  {
    optimizer = ens::L_BFGS(numBasis, maxIterations, armijoConstant, wolfe,
                            minGradientNorm, factr, maxLineSearchTrials);
  }
  PyL_BFGS(size_t numBasis, size_t maxIterations, double armijoConstant,
           double wolfe, double minGradientNorm, double factr,
           size_t maxLineSearchTrials, double minStep, double maxStep)
  {
    optimizer = ens::L_BFGS(numBasis, maxIterations, armijoConstant, wolfe,
                            minGradientNorm, factr, maxLineSearchTrials,
                            minStep, maxStep);
  }

  size_t getNumBasis() const { return optimizer.NumBasis(); }

  void setNumBasis(size_t numBasis) { optimizer.NumBasis() = numBasis; }

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }

  void setMaxIterations(size_t maxIterations)
  {
    optimizer.MaxIterations() = maxIterations;
  }

  double getArmijoConstant() const { return optimizer.ArmijoConstant(); }

  void setArmijoConstant(double armijoConstant)
  {
    optimizer.ArmijoConstant() = armijoConstant;
  }

  double getWolfe() const { return optimizer.Wolfe(); }

  void setWolfe(double wolfe) { optimizer.Wolfe() = wolfe; }

  double getMinGradientNorm() const { return optimizer.MinGradientNorm(); }

  void setMinGradientNorm(double minGradientNorm)
  {
    optimizer.MinGradientNorm() = minGradientNorm;
  }

  double getFactr() const { return optimizer.Factr(); }

  void setFactr(double factr) { optimizer.Factr() = factr; }

  size_t getMaxLineSearchTrials() const
  {
    return optimizer.MaxLineSearchTrials();
  }

  void setMaxLineSearchTrials(size_t maxLineSearchTrials)
  {
    optimizer.MaxLineSearchTrials() = maxLineSearchTrials;
  }

  double getMinStep() const { return optimizer.MinStep(); }

  void setMinStep(double minStep) { optimizer.MinStep() = minStep; }

  double getMaxStep() const { return optimizer.MaxStep(); }

  void setMaxStep(double maxStep) { optimizer.MaxStep() = maxStep; }

  py::array_t<double> Optimize(DifferentiableFunction f,
                               py::array_t<double> initial_point, ens::PyReport* report = nullptr)
  {
    py::buffer_info buf_info = initial_point.request();
    arma::vec arma_initial_point(static_cast<double *>(buf_info.ptr),
                                 buf_info.shape[0], false, true);

    DifferentiableFunctionWrapper fw(f);
    arma::vec result = arma_initial_point;
    if (report)
      optimizer.Optimize(fw, result, *report);
    else
      optimizer.Optimize(fw, result);
    return py::array_t<double>(result.n_elem, result.memptr());

  }

private:
  ens::L_BFGS optimizer;
};