#pragma once
#include <ensmallen.hpp>
#include "function_wrapper.hpp"
#include "constraints.hpp"

// Forward declarations for FrankWolfe types
using LpSolver = ens::FrankWolfe<ens::ConstrLpBallSolver, ens::UpdateClassic>;
using SimplexSolver = ens::FrankWolfe<ConstrSimplexSolver, ens::UpdateClassic>;

// Wrapper for FrankWolfe optimizer
class PyFrankWolfe
{
public:
  PyFrankWolfe(
      const std::string &constraint_type = "lp",
      double p = 2.0,
      size_t maxIterations = 100000,
      double tolerance = 1e-10)
  {
    if (constraint_type == "lp")
    {
      optimizer = std::make_unique<LpSolver>(
          ens::ConstrLpBallSolver(p),
          ens::UpdateClassic(),
          maxIterations,
          tolerance);
      simplex_optimizer = nullptr;
      using_simplex = false;
    }
    else if (constraint_type == "simplex")
    {
      optimizer = std::make_unique<SimplexSolver>(
          ConstrSimplexSolver(),
          ens::UpdateClassic(),
          maxIterations,
          tolerance);
      lp_optimizer = nullptr;
      using_simplex = true;
    }
    else
    {
      throw std::invalid_argument("constraint_type must be 'lp' or 'simplex'");
    }
  }

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }
  void setMaxIterations(size_t maxIterations)
  {
    optimizer.MaxIterations() = maxIterations;
  }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  py::array_t<double> Optimize(const DifferentiableFunction &f,
                               py::array_t<double> initial_point)
  {
    py::buffer_info buf_info = initial_point.request();
    arma::vec arma_initial_point(static_cast<double *>(buf_info.ptr),
                                 buf_info.shape[0], false, true);

    DifferentiableFunctionWrapper fw(f);
    arma::vec result = arma_initial_point;

    if (using_simplex)
      simplex_optimizer->Optimize(fw, result);
    else
      lp_optimizer->Optimize(fw, result);

    return py::array_t<double>(result.n_elem, result.memptr());
  }

private:
  std::unique_ptr<LpSolver> lp_optimizer;
  std::unique_ptr<SimplexSolver> simplex_optimizer;
  bool using_simplex;
};

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

  // Getters and setters
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
  ens::L_BFGS optimizer;
};

// Template class for Adam variants
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

  // Getters and setters
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
