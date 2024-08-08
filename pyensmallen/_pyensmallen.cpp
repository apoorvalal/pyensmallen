#include <armadillo>
#include <ensmallen.hpp>
#include <ensmallen_bits/adam/adam.hpp>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// Function type for optimization problems
using DifferentiableFunction =
    std::function<double(py::array_t<double>, py::array_t<double>)>;

// Wrapper class that implements the interface expected by ensmallen
class DifferentiableFunctionWrapper {
public:
  DifferentiableFunctionWrapper(const DifferentiableFunction &f) : f(f) {}

  double Evaluate(const arma::mat &parameters) {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    return f(py_params, py_grad);
  }

  void Gradient(const arma::mat &parameters, arma::mat &gradient) {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double *>(buf_info.ptr), parameters.n_rows,
                         parameters.n_cols);
  }

  double EvaluateWithGradient(const arma::mat &parameters,
                              arma::mat &gradient) {
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
                  const size_t batchSize) {
    return Evaluate(parameters);
  }
  void Gradient(const arma::mat &parameters, const size_t begin,
                arma::mat &gradient, const size_t batchSize) {
    Gradient(parameters, gradient);
  }
  double EvaluateWithGradient(const arma::mat &parameters, const size_t begin,
                              arma::mat &gradient, const size_t batchSize) {
    return EvaluateWithGradient(parameters, gradient);
  }
  size_t NumFunctions() const { return 1; }
  void Shuffle() {}

private:
  DifferentiableFunction f;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// New wrapper specifically for Frank-Wolfe
class FrankWolfeFunctionWrapper : public ens::FuncSq {
public:
  FrankWolfeFunctionWrapper(const DifferentiableFunction &f) : f(f) {}

  double Evaluate(const arma::mat &parameters) override {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    return f(py_params, py_grad);
  }

  void Gradient(const arma::mat &parameters, arma::mat &gradient) override {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double *>(buf_info.ptr), parameters.n_rows,
                         parameters.n_cols);
  }

  double EvaluateWithGradient(const arma::mat &parameters,
                              arma::mat &gradient) override {
    py::array_t<double> py_params(parameters.n_elem, parameters.memptr());
    py::array_t<double> py_grad(parameters.n_elem);
    double result = f(py_params, py_grad);
    py::buffer_info buf_info = py_grad.request();
    gradient = arma::mat(static_cast<double *>(buf_info.ptr), parameters.n_rows,
                         parameters.n_cols);
    return result;
  }

private:
  DifferentiableFunction f;
};

// Wrapper for FrankWolfe optimizer
class PyFrankWolfe {
public:
  PyFrankWolfe(double p = 2.0, size_t maxIterations = 100000,
               double tolerance = 1e-10)
      : optimizer(ens::ConstrLpBallSolver(p), ens::UpdateClassic(),
                  maxIterations, tolerance) {}

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }
  void setMaxIterations(size_t maxIterations) {
    optimizer.MaxIterations() = maxIterations;
  }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  py::array_t<double> Optimize(const DifferentiableFunction &f,
                               py::array_t<double> initial_point) {
    py::buffer_info buf_info = initial_point.request();
    arma::vec arma_initial_point(static_cast<double *>(buf_info.ptr),
                                 buf_info.shape[0], false, true);

    FrankWolfeFunctionWrapper fw(f);
    arma::vec result = arma_initial_point;

    optimizer.Optimize(fw, result);

    return py::array_t<double>(result.n_elem, result.memptr());
  }

private:
  ens::FrankWolfe<ens::ConstrLpBallSolver, ens::UpdateClassic> optimizer;
};

// Wrapper for OMP (Orthogonal Matching Pursuit) optimizer
class PyOMP {
public:
  PyOMP(double p = 2.0, size_t maxIterations = 100000, double tolerance = 1e-10)
      : optimizer(ens::ConstrLpBallSolver(p), ens::UpdateSpan(), maxIterations,
                  tolerance) {}

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }
  void setMaxIterations(size_t maxIterations) {
    optimizer.MaxIterations() = maxIterations;
  }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  py::array_t<double> Optimize(const DifferentiableFunction &f,
                               py::array_t<double> initial_point) {
    py::buffer_info buf_info = initial_point.request();
    arma::vec arma_initial_point(static_cast<double *>(buf_info.ptr),
                                 buf_info.shape[0], false, true);

    FrankWolfeFunctionWrapper fw(f);
    arma::vec result = arma_initial_point;

    optimizer.Optimize(fw, result);

    return py::array_t<double>(result.n_elem, result.memptr());
  }

private:
  ens::FrankWolfe<ens::ConstrLpBallSolver, ens::UpdateSpan> optimizer;
};

// Add this to your PYBIND11_MODULE
void init_frank_wolfe(py::module_ &m) {
  py::class_<PyFrankWolfe>(m, "FrankWolfe")
      .def(py::init<double, size_t, double>(), py::arg("p") = 2.0,
           py::arg("max_iterations") = 100000, py::arg("tolerance") = 1e-10)
      .def("get_max_iterations", &PyFrankWolfe::getMaxIterations)
      .def("set_max_iterations", &PyFrankWolfe::setMaxIterations)
      .def("get_tolerance", &PyFrankWolfe::getTolerance)
      .def("set_tolerance", &PyFrankWolfe::setTolerance)
      .def("optimize", &PyFrankWolfe::Optimize);

  py::class_<PyOMP>(m, "OMP")
      .def(py::init<double, size_t, double>(), py::arg("p") = 2.0,
           py::arg("max_iterations") = 100000, py::arg("tolerance") = 1e-10)
      .def("get_max_iterations", &PyOMP::getMaxIterations)
      .def("set_max_iterations", &PyOMP::setMaxIterations)
      .def("get_tolerance", &PyOMP::getTolerance)
      .def("set_tolerance", &PyOMP::setTolerance)
      .def("optimize", &PyOMP::Optimize);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// Wrapper for L-BFGS optimizer
class PyL_BFGS {
public:
  PyL_BFGS() : optimizer() {}
  PyL_BFGS(size_t numBasis, size_t maxIterations) {
    optimizer = ens::L_BFGS(numBasis, maxIterations);
  }
  PyL_BFGS(size_t numBasis, size_t maxIterations, double armijoConstant,
           double wolfe, double minGradientNorm, double factr,
           size_t maxLineSearchTrials) {
    optimizer = ens::L_BFGS(numBasis, maxIterations, armijoConstant, wolfe,
                            minGradientNorm, factr, maxLineSearchTrials);
  }
  PyL_BFGS(size_t numBasis, size_t maxIterations, double armijoConstant,
           double wolfe, double minGradientNorm, double factr,
           size_t maxLineSearchTrials, double minStep, double maxStep) {
    optimizer = ens::L_BFGS(numBasis, maxIterations, armijoConstant, wolfe,
                            minGradientNorm, factr, maxLineSearchTrials,
                            minStep, maxStep);
  }

  size_t getNumBasis() const { return optimizer.NumBasis(); }

  void setNumBasis(size_t numBasis) { optimizer.NumBasis() = numBasis; }

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }

  void setMaxIterations(size_t maxIterations) {
    optimizer.MaxIterations() = maxIterations;
  }

  double getArmijoConstant() const { return optimizer.ArmijoConstant(); }

  void setArmijoConstant(double armijoConstant) {
    optimizer.ArmijoConstant() = armijoConstant;
  }

  double getWolfe() const { return optimizer.Wolfe(); }

  void setWolfe(double wolfe) { optimizer.Wolfe() = wolfe; }

  double getMinGradientNorm() const { return optimizer.MinGradientNorm(); }

  void setMinGradientNorm(double minGradientNorm) {
    optimizer.MinGradientNorm() = minGradientNorm;
  }

  double getFactr() const { return optimizer.Factr(); }

  void setFactr(double factr) { optimizer.Factr() = factr; }

  size_t getMaxLineSearchTrials() const {
    return optimizer.MaxLineSearchTrials();
  }

  void setMaxLineSearchTrials(size_t maxLineSearchTrials) {
    optimizer.MaxLineSearchTrials() = maxLineSearchTrials;
  }

  double getMinStep() const { return optimizer.MinStep(); }

  void setMinStep(double minStep) { optimizer.MinStep() = minStep; }

  double getMaxStep() const { return optimizer.MaxStep(); }

  void setMaxStep(double maxStep) { optimizer.MaxStep() = maxStep; }

  py::array_t<double> Optimize(DifferentiableFunction f,
                               py::array_t<double> initial_point) {
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
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// Wrapper for Adam optimizer
template <typename UpdateRule> class PyAdamType {
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
  void setMaxIterations(size_t maxIterations) {
    optimizer.MaxIterations() = maxIterations;
  }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  bool getShuffle() const { return optimizer.Shuffle(); }
  void setShuffle(bool shuffle) { optimizer.Shuffle() = shuffle; }

  bool getExactObjective() const { return optimizer.ExactObjective(); }
  void setExactObjective(bool exactObjective) {
    optimizer.ExactObjective() = exactObjective;
  }

  bool getResetPolicy() const { return optimizer.ResetPolicy(); }
  void setResetPolicy(bool resetPolicy) {
    optimizer.ResetPolicy() = resetPolicy;
  }

  py::array_t<double> Optimize(DifferentiableFunction f,
                               py::array_t<double> initial_point) {
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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(_pyensmallen, m) {
  py::class_<PyL_BFGS>(m, "L_BFGS")
      .def(py::init<>())
      .def(py::init<size_t, size_t>(), py::arg("numBasis"),
           py::arg("maxIterations"))
      .def(py::init<size_t, size_t, double, double, double, double, size_t>(),
           py::arg("numBasis"), py::arg("maxIterations"),
           py::arg("armijoConstant"), py::arg("wolfe"),
           py::arg("minGradientNorm"), py::arg("factr"),
           py::arg("maxLineSearchTrials"))
      .def(py::init<size_t, size_t, double, double, double, double, size_t,
                    double, double>(),
           py::arg("numBasis"), py::arg("maxIterations"),
           py::arg("armijoConstant"), py::arg("wolfe"),
           py::arg("minGradientNorm"), py::arg("factr"),
           py::arg("maxLineSearchTrials"), py::arg("minStep"),
           py::arg("maxStep"))
      .def_property("numBasis", &PyL_BFGS::getNumBasis, &PyL_BFGS::setNumBasis)
      .def_property("maxIterations", &PyL_BFGS::getMaxIterations,
                    &PyL_BFGS::setMaxIterations)
      .def_property("armijoConstant", &PyL_BFGS::getArmijoConstant,
                    &PyL_BFGS::setArmijoConstant)
      .def_property("wolfe", &PyL_BFGS::getWolfe, &PyL_BFGS::setWolfe)
      .def_property("minGradientNorm", &PyL_BFGS::getMinGradientNorm,
                    &PyL_BFGS::setMinGradientNorm)
      .def_property("factr", &PyL_BFGS::getFactr, &PyL_BFGS::setFactr)
      .def_property("maxLineSearchTrials", &PyL_BFGS::getMaxLineSearchTrials,
                    &PyL_BFGS::setMaxLineSearchTrials)
      .def_property("minStep", &PyL_BFGS::getMinStep, &PyL_BFGS::setMinStep)
      .def_property("maxStep", &PyL_BFGS::getMaxStep, &PyL_BFGS::setMaxStep)
      .def("optimize", &PyL_BFGS::Optimize);
  // frank wolfe
  init_frank_wolfe(int &m);
  // Adam
  py::class_<PyAdamType<ens::AdamUpdate>>(m, "Adam")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false)
      .def_property("stepSize", &PyAdamType<ens::AdamUpdate>::getStepSize,
                    &PyAdamType<ens::AdamUpdate>::setStepSize)
      .def_property("batchSize", &PyAdamType<ens::AdamUpdate>::getBatchSize,
                    &PyAdamType<ens::AdamUpdate>::setBatchSize)
      .def_property("beta1", &PyAdamType<ens::AdamUpdate>::getBeta1,
                    &PyAdamType<ens::AdamUpdate>::setBeta1)
      .def_property("beta2", &PyAdamType<ens::AdamUpdate>::getBeta2,
                    &PyAdamType<ens::AdamUpdate>::setBeta2)
      .def_property("epsilon", &PyAdamType<ens::AdamUpdate>::getEpsilon,
                    &PyAdamType<ens::AdamUpdate>::setEpsilon)
      .def_property("maxIterations",
                    &PyAdamType<ens::AdamUpdate>::getMaxIterations,
                    &PyAdamType<ens::AdamUpdate>::setMaxIterations)
      .def_property("tolerance", &PyAdamType<ens::AdamUpdate>::getTolerance,
                    &PyAdamType<ens::AdamUpdate>::setTolerance)
      .def_property("shuffle", &PyAdamType<ens::AdamUpdate>::getShuffle,
                    &PyAdamType<ens::AdamUpdate>::setShuffle)
      .def_property("exactObjective",
                    &PyAdamType<ens::AdamUpdate>::getExactObjective,
                    &PyAdamType<ens::AdamUpdate>::setExactObjective)
      .def_property("resetPolicy", &PyAdamType<ens::AdamUpdate>::getResetPolicy,
                    &PyAdamType<ens::AdamUpdate>::setResetPolicy)
      .def("optimize", &PyAdamType<ens::AdamUpdate>::Optimize);
  // AdaMax
  py::class_<PyAdamType<ens::AdaMaxUpdate>>(m, "AdaMax")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false)
      .def_property("stepSize", &PyAdamType<ens::AdaMaxUpdate>::getStepSize,
                    &PyAdamType<ens::AdaMaxUpdate>::setStepSize)
      .def_property("batchSize", &PyAdamType<ens::AdaMaxUpdate>::getBatchSize,
                    &PyAdamType<ens::AdaMaxUpdate>::setBatchSize)
      .def_property("beta1", &PyAdamType<ens::AdaMaxUpdate>::getBeta1,
                    &PyAdamType<ens::AdaMaxUpdate>::setBeta1)
      .def_property("beta2", &PyAdamType<ens::AdaMaxUpdate>::getBeta2,
                    &PyAdamType<ens::AdaMaxUpdate>::setBeta2)
      .def_property("epsilon", &PyAdamType<ens::AdaMaxUpdate>::getEpsilon,
                    &PyAdamType<ens::AdaMaxUpdate>::setEpsilon)
      .def_property("maxIterations",
                    &PyAdamType<ens::AdaMaxUpdate>::getMaxIterations,
                    &PyAdamType<ens::AdaMaxUpdate>::setMaxIterations)
      .def_property("tolerance", &PyAdamType<ens::AdaMaxUpdate>::getTolerance,
                    &PyAdamType<ens::AdaMaxUpdate>::setTolerance)
      .def_property("shuffle", &PyAdamType<ens::AdaMaxUpdate>::getShuffle,
                    &PyAdamType<ens::AdaMaxUpdate>::setShuffle)
      .def_property("exactObjective",
                    &PyAdamType<ens::AdaMaxUpdate>::getExactObjective,
                    &PyAdamType<ens::AdaMaxUpdate>::setExactObjective)
      .def_property("resetPolicy",
                    &PyAdamType<ens::AdaMaxUpdate>::getResetPolicy,
                    &PyAdamType<ens::AdaMaxUpdate>::setResetPolicy)
      .def("optimize", &PyAdamType<ens::AdaMaxUpdate>::Optimize);
  // AMSGrad
  py::class_<PyAdamType<ens::AMSGradUpdate>>(m, "AMSGrad")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false)
      .def_property("stepSize", &PyAdamType<ens::AMSGradUpdate>::getStepSize,
                    &PyAdamType<ens::AMSGradUpdate>::setStepSize)
      .def_property("batchSize", &PyAdamType<ens::AMSGradUpdate>::getBatchSize,
                    &PyAdamType<ens::AMSGradUpdate>::setBatchSize)
      .def_property("beta1", &PyAdamType<ens::AMSGradUpdate>::getBeta1,
                    &PyAdamType<ens::AMSGradUpdate>::setBeta1)
      .def_property("beta2", &PyAdamType<ens::AMSGradUpdate>::getBeta2,
                    &PyAdamType<ens::AMSGradUpdate>::setBeta2)
      .def_property("epsilon", &PyAdamType<ens::AMSGradUpdate>::getEpsilon,
                    &PyAdamType<ens::AMSGradUpdate>::setEpsilon)
      .def_property("maxIterations",
                    &PyAdamType<ens::AMSGradUpdate>::getMaxIterations,
                    &PyAdamType<ens::AMSGradUpdate>::setMaxIterations)
      .def_property("tolerance", &PyAdamType<ens::AMSGradUpdate>::getTolerance,
                    &PyAdamType<ens::AMSGradUpdate>::setTolerance)
      .def_property("shuffle", &PyAdamType<ens::AMSGradUpdate>::getShuffle,
                    &PyAdamType<ens::AMSGradUpdate>::setShuffle)
      .def_property("exactObjective",
                    &PyAdamType<ens::AMSGradUpdate>::getExactObjective,
                    &PyAdamType<ens::AMSGradUpdate>::setExactObjective)
      .def_property("resetPolicy",
                    &PyAdamType<ens::AMSGradUpdate>::getResetPolicy,
                    &PyAdamType<ens::AMSGradUpdate>::setResetPolicy)
      .def("optimize", &PyAdamType<ens::AMSGradUpdate>::Optimize);
  // OptimisticAdam
  py::class_<PyAdamType<ens::OptimisticAdamUpdate>>(m, "OptimisticAdam")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false)
      .def_property("stepSize",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getStepSize,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setStepSize)
      .def_property("batchSize",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getBatchSize,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setBatchSize)
      .def_property("beta1", &PyAdamType<ens::OptimisticAdamUpdate>::getBeta1,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setBeta1)
      .def_property("beta2", &PyAdamType<ens::OptimisticAdamUpdate>::getBeta2,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setBeta2)
      .def_property("epsilon",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getEpsilon,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setEpsilon)
      .def_property("maxIterations",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getMaxIterations,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setMaxIterations)
      .def_property("tolerance",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getTolerance,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setTolerance)
      .def_property("shuffle",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getShuffle,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setShuffle)
      .def_property("exactObjective",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getExactObjective,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setExactObjective)
      .def_property("resetPolicy",
                    &PyAdamType<ens::OptimisticAdamUpdate>::getResetPolicy,
                    &PyAdamType<ens::OptimisticAdamUpdate>::setResetPolicy)
      .def("optimize", &PyAdamType<ens::OptimisticAdamUpdate>::Optimize);
  // Nadam
  py::class_<PyAdamType<ens::NadamUpdate>>(m, "Nadam")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false)
      .def_property("stepSize", &PyAdamType<ens::NadamUpdate>::getStepSize,
                    &PyAdamType<ens::NadamUpdate>::setStepSize)
      .def_property("batchSize", &PyAdamType<ens::NadamUpdate>::getBatchSize,
                    &PyAdamType<ens::NadamUpdate>::setBatchSize)
      .def_property("beta1", &PyAdamType<ens::NadamUpdate>::getBeta1,
                    &PyAdamType<ens::NadamUpdate>::setBeta1)
      .def_property("beta2", &PyAdamType<ens::NadamUpdate>::getBeta2,
                    &PyAdamType<ens::NadamUpdate>::setBeta2)
      .def_property("epsilon", &PyAdamType<ens::NadamUpdate>::getEpsilon,
                    &PyAdamType<ens::NadamUpdate>::setEpsilon)
      .def_property("maxIterations",
                    &PyAdamType<ens::NadamUpdate>::getMaxIterations,
                    &PyAdamType<ens::NadamUpdate>::setMaxIterations)
      .def_property("tolerance", &PyAdamType<ens::NadamUpdate>::getTolerance,
                    &PyAdamType<ens::NadamUpdate>::setTolerance)
      .def_property("shuffle", &PyAdamType<ens::NadamUpdate>::getShuffle,
                    &PyAdamType<ens::NadamUpdate>::setShuffle)
      .def_property("exactObjective",
                    &PyAdamType<ens::NadamUpdate>::getExactObjective,
                    &PyAdamType<ens::NadamUpdate>::setExactObjective)
      .def_property("resetPolicy",
                    &PyAdamType<ens::NadamUpdate>::getResetPolicy,
                    &PyAdamType<ens::NadamUpdate>::setResetPolicy)
      .def("optimize", &PyAdamType<ens::NadamUpdate>::Optimize);
}
