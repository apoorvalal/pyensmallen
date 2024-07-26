#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <ensmallen.hpp>
#include <armadillo>

namespace py = pybind11;

// Function type for optimization problems
using OptimizationFunction = std::function<double(py::array_t<double>, py::array_t<double>)>;

// Wrapper class that implements the interface expected by ensmallen
class FunctionWrapper
{
 public:
  FunctionWrapper(const OptimizationFunction& f) : f(f) {}

  double Evaluate(const arma::mat& parameters)
  {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    return f(py_params, py_grad);
  }

  void Gradient(const arma::mat& parameters, arma::mat& gradient)
  {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double*>(buf_info.ptr), parameters.n_rows, parameters.n_cols);
  }

  double EvaluateWithGradient(const arma::mat& parameters, arma::mat& gradient)
  {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    double result = f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double*>(buf_info.ptr), parameters.n_rows, parameters.n_cols);
    return result;
  }

 private:
  OptimizationFunction f;
};

// Wrapper for L-BFGS optimizer
class PyL_BFGS {
public:
    PyL_BFGS() : optimizer() {}

    py::array_t<double> Optimize(OptimizationFunction f, py::array_t<double> initial_point) {
        py::buffer_info buf_info = initial_point.request();
        arma::vec arma_initial_point(static_cast<double*>(buf_info.ptr), buf_info.shape[0], false, true);

        FunctionWrapper fw(f);
        arma::vec result = arma_initial_point;

        optimizer.Optimize(fw, result);

        return py::array_t<double>(result.n_elem, result.memptr());
    }

private:
    ens::L_BFGS optimizer;
};

PYBIND11_MODULE(_pyensmallen, m) {
    py::class_<PyL_BFGS>(m, "L_BFGS")
        .def(py::init<>())
        .def("optimize", &PyL_BFGS::Optimize);
}
