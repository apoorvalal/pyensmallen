#pragma once

#include "utils.hpp"

// First, define the simplex solver in the ens namespace
namespace ens
{
  class ConstrSimplexSolver
  {
  public:
    ConstrSimplexSolver() {}

    template <typename MatType>
    void Optimize(const MatType &v, MatType &s)
    {
      typedef typename MatType::elem_type ElemType;
      s.zeros(v.n_elem);
      arma::uword k = 0;
      v.min(k);
      s(k) = 1.0;
      return;
    }
  };
} // namespace ens

// Wrapper for FrankWolfe optimizer with L_p ball constraints
class PyFrankWolfe
{
public:
  PyFrankWolfe(double p = 2.0,
               size_t maxIterations = 100000,
               double tolerance = 1e-10)
      : optimizer(
            ens::ConstrLpBallSolver(p),
            ens::UpdateClassic(),
            maxIterations,
            tolerance) {}

  // Constructor with explicit lambda
  PyFrankWolfe(double p,
               const py::array_t<double> &lambda,
               size_t maxIterations = 100000,
               double tolerance = 1e-10)
      : optimizer(
            ens::ConstrLpBallSolver(p,
                                    arma::vec(static_cast<double *>(lambda.request().ptr),
                                              lambda.request().size)),
            ens::UpdateClassic(),
            maxIterations,
            tolerance) {}

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

    optimizer.Optimize(fw, result);

    return py::array_t<double>(result.n_elem, result.memptr());
  }

private:
  ens::FrankWolfe<ens::ConstrLpBallSolver, ens::UpdateClassic> optimizer;
};

// Wrapper for FrankWolfe optimizer with simplex constraints
class PySimplexFrankWolfe
{
public:
  PySimplexFrankWolfe(size_t maxIterations = 100000,
                      double tolerance = 1e-10)
      : optimizer(
            ens::ConstrSimplexSolver(),
            ens::UpdateClassic(),
            maxIterations,
            tolerance) {}

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

    optimizer.Optimize(fw, result);

    return py::array_t<double>(result.n_elem, result.memptr());
  }

private:
  ens::FrankWolfe<ens::ConstrSimplexSolver, ens::UpdateClassic> optimizer;
};