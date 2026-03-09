#include <ensmallen.hpp>
#include <pybind11/pybind11.h>

#include "utils.hpp"
#include "newton_type.hpp"
#include "constrained.hpp"
#include "first_order.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyensmallen, m)
{
  const char* lbfgs_doc =
      "Limited-memory BFGS optimizer for smooth objectives.\n\n"
      "Use this optimizer for differentiable full-batch objectives where a\n"
      "quasi-Newton method is appropriate. The objective callable must accept\n"
      "a parameter vector and a writable gradient vector, and return the scalar\n"
      "objective value.";
  const char* fw_doc =
      "Frank-Wolfe optimizer over an L_p ball constraint.\n\n"
      "This optimizer is useful for smooth constrained problems, including\n"
      "ridge-like and lasso-like feasible regions expressed as norm balls.";
  const char* simplex_fw_doc =
      "Frank-Wolfe optimizer specialized to simplex constraints.\n\n"
      "Use this optimizer when parameters must be nonnegative and sum to one,\n"
      "for example in balancing-weight or synthetic-control style problems.";
  const char* adam_doc =
      "First-order Adam-family optimizer for differentiable objectives.\n\n"
      "These optimizers are exposed through the current full-batch objective\n"
      "wrapper. The callable must accept a parameter vector and a writable\n"
      "gradient vector, and return the scalar objective value.";
  const char* optimize_doc =
      "Optimize an objective from the provided starting point.\n\n"
      "Parameters\n"
      "----------\n"
      "objective : callable\n"
      "    Callable with signature ``objective(params, gradient)``. The callable\n"
      "    should write the gradient in place and return the scalar objective.\n"
      "initial_point : ndarray\n"
      "    Initial parameter vector.\n\n"
      "Returns\n"
      "-------\n"
      "ndarray\n"
      "    Optimized parameter vector.";

  // L-BFGS (Newton-type) optimizer
  py::class_<PyL_BFGS>(m, "L_BFGS", lbfgs_doc)
      .def(py::init<>(), "Construct an L-BFGS optimizer with default settings.")
      .def(py::init<size_t, size_t>(), py::arg("numBasis"),
           py::arg("maxIterations"),
           "Construct an L-BFGS optimizer with explicit memory and iteration limits.")
      .def(py::init<size_t, size_t, double, double, double, double, size_t>(),
           py::arg("numBasis"), py::arg("maxIterations"),
           py::arg("armijoConstant"), py::arg("wolfe"),
           py::arg("minGradientNorm"), py::arg("factr"),
           py::arg("maxLineSearchTrials"),
           "Construct an L-BFGS optimizer with explicit line-search controls.")
      .def(py::init<size_t, size_t, double, double, double, double, size_t,
                    double, double>(),
           py::arg("numBasis"), py::arg("maxIterations"),
           py::arg("armijoConstant"), py::arg("wolfe"),
           py::arg("minGradientNorm"), py::arg("factr"),
           py::arg("maxLineSearchTrials"), py::arg("minStep"),
           py::arg("maxStep"),
           "Construct an L-BFGS optimizer with explicit step-size bounds.")
      .def_property("numBasis", &PyL_BFGS::getNumBasis, &PyL_BFGS::setNumBasis,
                    "Number of correction vectors retained in memory.")
      .def_property("maxIterations", &PyL_BFGS::getMaxIterations,
                    &PyL_BFGS::setMaxIterations,
                    "Maximum number of optimizer iterations.")
      .def_property("armijoConstant", &PyL_BFGS::getArmijoConstant,
                    &PyL_BFGS::setArmijoConstant,
                    "Armijo line-search constant.")
      .def_property("wolfe", &PyL_BFGS::getWolfe, &PyL_BFGS::setWolfe,
                    "Wolfe curvature constant.")
      .def_property("minGradientNorm", &PyL_BFGS::getMinGradientNorm,
                    &PyL_BFGS::setMinGradientNorm,
                    "Termination tolerance based on gradient norm.")
      .def_property("factr", &PyL_BFGS::getFactr, &PyL_BFGS::setFactr,
                    "Relative objective tolerance used by the optimizer.")
      .def_property("maxLineSearchTrials", &PyL_BFGS::getMaxLineSearchTrials,
                    &PyL_BFGS::setMaxLineSearchTrials,
                    "Maximum number of line-search attempts per iteration.")
      .def_property("minStep", &PyL_BFGS::getMinStep, &PyL_BFGS::setMinStep,
                    "Lower bound on the line-search step size.")
      .def_property("maxStep", &PyL_BFGS::getMaxStep, &PyL_BFGS::setMaxStep,
                    "Upper bound on the line-search step size.")
      .def("optimize", &PyL_BFGS::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // FrankWolfe - constrained optimization
  py::class_<PyFrankWolfe>(m, "FrankWolfe", fw_doc)
      .def(py::init<double, size_t, double>(),
           py::arg("p") = 2.0,
           py::arg("max_iterations") = 100000,
           py::arg("tolerance") = 1e-10,
           "Construct a Frank-Wolfe optimizer over an L_p ball.")
      .def(py::init<double, py::array_t<double>, size_t, double>(),
           py::arg("p"),
           py::arg("lambda"),
           py::arg("max_iterations") = 100000,
           py::arg("tolerance") = 1e-10,
           "Construct a Frank-Wolfe optimizer with coordinate-wise radius limits.")
      .def("get_max_iterations", &PyFrankWolfe::getMaxIterations,
           "Return the maximum number of iterations.")
      .def("set_max_iterations", &PyFrankWolfe::setMaxIterations,
           py::arg("max_iterations"),
           "Set the maximum number of iterations.")
      .def("get_tolerance", &PyFrankWolfe::getTolerance,
           "Return the optimization tolerance.")
      .def("set_tolerance", &PyFrankWolfe::setTolerance, py::arg("tolerance"),
           "Set the optimization tolerance.")
      .def("optimize", &PyFrankWolfe::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // SimplexFrankWolfe - simplex constrained optimization
  py::class_<PySimplexFrankWolfe>(m, "SimplexFrankWolfe", simplex_fw_doc)
      .def(py::init<size_t, double>(),
           py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-10,
           "Construct a simplex-constrained Frank-Wolfe optimizer.")
      .def_property("maxIterations",
                    &PySimplexFrankWolfe::getMaxIterations,
                    &PySimplexFrankWolfe::setMaxIterations,
                    "Maximum number of optimizer iterations.")
      .def_property("tolerance",
                    &PySimplexFrankWolfe::getTolerance,
                    &PySimplexFrankWolfe::setTolerance,
                    "Optimization tolerance.")
      .def("optimize", &PySimplexFrankWolfe::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // Adam - first-order optimization
  py::class_<PyAdamType<ens::AdamUpdate>>(m, "Adam", adam_doc)
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false,
           "Construct an Adam optimizer.")
      .def_property("stepSize", &PyAdamType<ens::AdamUpdate>::getStepSize,
                    &PyAdamType<ens::AdamUpdate>::setStepSize,
                    "Learning rate.")
      .def_property("batchSize", &PyAdamType<ens::AdamUpdate>::getBatchSize,
                    &PyAdamType<ens::AdamUpdate>::setBatchSize,
                    "Nominal batch size parameter exposed by ensmallen.")
      .def_property("beta1", &PyAdamType<ens::AdamUpdate>::getBeta1,
                    &PyAdamType<ens::AdamUpdate>::setBeta1,
                    "Exponential decay for first moments.")
      .def_property("beta2", &PyAdamType<ens::AdamUpdate>::getBeta2,
                    &PyAdamType<ens::AdamUpdate>::setBeta2,
                    "Exponential decay for second moments.")
      .def_property("epsilon", &PyAdamType<ens::AdamUpdate>::getEpsilon,
                    &PyAdamType<ens::AdamUpdate>::setEpsilon,
                    "Numerical stabilization constant.")
      .def_property("maxIterations",
                    &PyAdamType<ens::AdamUpdate>::getMaxIterations,
                    &PyAdamType<ens::AdamUpdate>::setMaxIterations,
                    "Maximum number of iterations.")
      .def_property("tolerance", &PyAdamType<ens::AdamUpdate>::getTolerance,
                    &PyAdamType<ens::AdamUpdate>::setTolerance,
                    "Termination tolerance.")
      .def_property("shuffle", &PyAdamType<ens::AdamUpdate>::getShuffle,
                    &PyAdamType<ens::AdamUpdate>::setShuffle,
                    "Whether to shuffle separable objectives between epochs.")
      .def_property("exactObjective",
                    &PyAdamType<ens::AdamUpdate>::getExactObjective,
                    &PyAdamType<ens::AdamUpdate>::setExactObjective,
                    "Whether to evaluate the exact objective after optimization.")
      .def_property("resetPolicy", &PyAdamType<ens::AdamUpdate>::getResetPolicy,
                    &PyAdamType<ens::AdamUpdate>::setResetPolicy,
                    "Whether optimizer state is reset before each call.")
      .def("optimize", &PyAdamType<ens::AdamUpdate>::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // AdaMax - Adam variant
  py::class_<PyAdamType<ens::AdaMaxUpdate>>(m, "AdaMax",
      "AdaMax optimizer, a max-norm variant of Adam.\n\nUse the same objective interface as Adam.")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false,
           "Construct an AdaMax optimizer.")
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
      .def("optimize", &PyAdamType<ens::AdaMaxUpdate>::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // AMSGrad - Adam variant
  py::class_<PyAdamType<ens::AMSGradUpdate>>(m, "AMSGrad",
      "AMSGrad optimizer, a monotone-variance variant of Adam.\n\nUse the same objective interface as Adam.")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false,
           "Construct an AMSGrad optimizer.")
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
      .def("optimize", &PyAdamType<ens::AMSGradUpdate>::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // OptimisticAdam - Adam variant
  py::class_<PyAdamType<ens::OptimisticAdamUpdate>>(m, "OptimisticAdam",
      "Optimistic Adam optimizer.\n\nUse the same objective interface as Adam.")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false,
           "Construct an OptimisticAdam optimizer.")
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
      .def("optimize", &PyAdamType<ens::OptimisticAdamUpdate>::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
  
  // Nadam - Adam variant
  py::class_<PyAdamType<ens::NadamUpdate>>(m, "Nadam",
      "Nadam optimizer.\n\nUse the same objective interface as Adam.")
      .def(py::init<double, size_t, double, double, double, size_t, double,
                    bool, bool, bool>(),
           py::arg("stepSize") = 0.001, py::arg("batchSize") = 32,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("eps") = 1e-8, py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5, py::arg("shuffle") = true,
           py::arg("resetPolicy") = true, py::arg("exactObjective") = false,
           "Construct a Nadam optimizer.")
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
      .def("optimize", &PyAdamType<ens::NadamUpdate>::Optimize,
           py::arg("objective"), py::arg("initial_point"), optimize_doc);
}
