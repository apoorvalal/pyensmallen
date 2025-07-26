#include <ensmallen.hpp>
#include <pybind11/pybind11.h>

#include "utils.hpp"
#include "newton_type.hpp"
#include "constrained.hpp"
#include "first_order.hpp"
#include "additional_optimizers.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyensmallen, m)
{
  // L-BFGS (Newton-type) optimizer
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
  
  // FrankWolfe - constrained optimization
  py::class_<PyFrankWolfe>(m, "FrankWolfe")
      .def(py::init<double, size_t, double>(),
           py::arg("p") = 2.0,
           py::arg("max_iterations") = 100000,
           py::arg("tolerance") = 1e-10)
      .def(py::init<double, py::array_t<double>, size_t, double>(),
           py::arg("p"),
           py::arg("lambda"),
           py::arg("max_iterations") = 100000,
           py::arg("tolerance") = 1e-10)
      .def("get_max_iterations", &PyFrankWolfe::getMaxIterations)
      .def("set_max_iterations", &PyFrankWolfe::setMaxIterations)
      .def("get_tolerance", &PyFrankWolfe::getTolerance)
      .def("set_tolerance", &PyFrankWolfe::setTolerance)
      .def("optimize", &PyFrankWolfe::Optimize);
  
  // SimplexFrankWolfe - simplex constrained optimization
  py::class_<PySimplexFrankWolfe>(m, "SimplexFrankWolfe")
      .def(py::init<size_t, double>(),
           py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-10)
      .def_property("maxIterations",
                    &PySimplexFrankWolfe::getMaxIterations,
                    &PySimplexFrankWolfe::setMaxIterations)
      .def_property("tolerance",
                    &PySimplexFrankWolfe::getTolerance,
                    &PySimplexFrankWolfe::setTolerance)
      .def("optimize", &PySimplexFrankWolfe::Optimize);
  
  // Adam - first-order optimization
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
  
  // AdaMax - Adam variant
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
  
  // AMSGrad - Adam variant
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
  
  // OptimisticAdam - Adam variant
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
  
  // Nadam - Adam variant
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

  // Coordinate Descent optimizers
  py::class_<PyCyclicDescent>(m, "CyclicDescent")
      .def(py::init<double, size_t, double, size_t>(),
           py::arg("stepSize") = 0.01,
           py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5,
           py::arg("updateInterval") = 1000)
      .def_property("stepSize", &PyCyclicDescent::getStepSize,
                    &PyCyclicDescent::setStepSize)
      .def_property("maxIterations", &PyCyclicDescent::getMaxIterations,
                    &PyCyclicDescent::setMaxIterations)
      .def_property("tolerance", &PyCyclicDescent::getTolerance,
                    &PyCyclicDescent::setTolerance)
      .def_property("updateInterval", &PyCyclicDescent::getUpdateInterval,
                    &PyCyclicDescent::setUpdateInterval)
      .def("optimize", &PyCyclicDescent::Optimize<PyObjectiveFunction>);

  py::class_<PyRandomDescent>(m, "RandomDescent")
      .def(py::init<double, size_t, double, size_t>(),
           py::arg("stepSize") = 0.01,
           py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5,
           py::arg("updateInterval") = 1000)
      .def_property("stepSize", &PyRandomDescent::getStepSize,
                    &PyRandomDescent::setStepSize)
      .def_property("maxIterations", &PyRandomDescent::getMaxIterations,
                    &PyRandomDescent::setMaxIterations)
      .def_property("tolerance", &PyRandomDescent::getTolerance,
                    &PyRandomDescent::setTolerance)
      .def_property("updateInterval", &PyRandomDescent::getUpdateInterval,
                    &PyRandomDescent::setUpdateInterval)
      .def("optimize", &PyRandomDescent::Optimize<PyObjectiveFunction>);

  py::class_<PyGreedyDescent>(m, "GreedyDescent")
      .def(py::init<double, size_t, double, size_t>(),
           py::arg("stepSize") = 0.01,
           py::arg("maxIterations") = 100000,
           py::arg("tolerance") = 1e-5,
           py::arg("updateInterval") = 1000)
      .def_property("stepSize", &PyGreedyDescent::getStepSize,
                    &PyGreedyDescent::setStepSize)
      .def_property("maxIterations", &PyGreedyDescent::getMaxIterations,
                    &PyGreedyDescent::setMaxIterations)
      .def_property("tolerance", &PyGreedyDescent::getTolerance,
                    &PyGreedyDescent::setTolerance)
      .def_property("updateInterval", &PyGreedyDescent::getUpdateInterval,
                    &PyGreedyDescent::setUpdateInterval)
      .def("optimize", &PyGreedyDescent::Optimize<PyObjectiveFunction>);

  // Simulated Annealing optimizer
  py::class_<PySimulatedAnnealingDefault>(m, "SimulatedAnnealing")
      .def(py::init<size_t, double, size_t, size_t, double, size_t, double, double, double>(),
           py::arg("maxIterations") = 1000000,
           py::arg("initT") = 10000.0,
           py::arg("initMoves") = 1000,
           py::arg("moveCtrlSweep") = 100,
           py::arg("tolerance") = 1e-5,
           py::arg("maxToleranceSweep") = 3,
           py::arg("maxMoveCoef") = 20,
           py::arg("initMoveCoef") = 0.3,
           py::arg("gain") = 0.3)
      .def_property("maxIterations", &PySimulatedAnnealingDefault::getMaxIterations,
                    &PySimulatedAnnealingDefault::setMaxIterations)
      .def_property("temperature", &PySimulatedAnnealingDefault::getTemperature,
                    &PySimulatedAnnealingDefault::setTemperature)
      .def_property("initMoves", &PySimulatedAnnealingDefault::getInitMoves,
                    &PySimulatedAnnealingDefault::setInitMoves)
      .def_property("moveCtrlSweep", &PySimulatedAnnealingDefault::getMoveCtrlSweep,
                    &PySimulatedAnnealingDefault::setMoveCtrlSweep)
      .def_property("tolerance", &PySimulatedAnnealingDefault::getTolerance,
                    &PySimulatedAnnealingDefault::setTolerance)
      .def_property("maxToleranceSweep", &PySimulatedAnnealingDefault::getMaxToleranceSweep,
                    &PySimulatedAnnealingDefault::setMaxToleranceSweep)
      .def_property("gain", &PySimulatedAnnealingDefault::getGain,
                    &PySimulatedAnnealingDefault::setGain)
      .def("optimize", &PySimulatedAnnealingDefault::Optimize<PyObjectiveFunction>);
}