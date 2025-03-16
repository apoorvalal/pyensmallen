#pragma once

#include <armadillo>
#include <ensmallen.hpp>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Function type for optimization problems
using DifferentiableFunction = std::function<double(py::array_t<double>, py::array_t<double>)>;

// Wrapper class that implements the interface expected by ensmallen
class DifferentiableFunctionWrapper
{
public:
  DifferentiableFunctionWrapper(const DifferentiableFunction &f) : f(f) {}

  double Evaluate(const arma::mat &parameters)
  {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    return f(py_params, py_grad);
  }

  void Gradient(const arma::mat &parameters, arma::mat &gradient)
  {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double *>(buf_info.ptr), parameters.n_rows,
                        parameters.n_cols);
  }

  double EvaluateWithGradient(const arma::mat &parameters,
                            arma::mat &gradient)
  {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    double result = f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double *>(buf_info.ptr), parameters.n_rows,
                        parameters.n_cols);
    return result;
  }
  
  // Separable versions
  double Evaluate(const arma::mat &parameters, const size_t begin,
                const size_t batchSize)
  {
    return Evaluate(parameters);
  }
  
  void Gradient(const arma::mat &parameters, const size_t begin,
              arma::mat &gradient, const size_t batchSize)
  {
    Gradient(parameters, gradient);
  }
  
  double EvaluateWithGradient(const arma::mat &parameters, const size_t begin,
                            arma::mat &gradient, const size_t batchSize)
  {
    return EvaluateWithGradient(parameters, gradient);
  }
  
  size_t NumFunctions() const { return 1; }
  void Shuffle() {}

private:
  DifferentiableFunction f;
};