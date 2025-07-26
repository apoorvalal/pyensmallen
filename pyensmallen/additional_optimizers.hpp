#pragma once

#include "utils.hpp"

// Wrapper for Coordinate Descent optimizer
template<typename DescentPolicy = ens::RandomDescent>
class PyCoordinateDescent
{
public:
  PyCoordinateDescent(double stepSize = 0.01,
                     size_t maxIterations = 100000,
                     double tolerance = 1e-5,
                     size_t updateInterval = 1000)
      : optimizer(stepSize, maxIterations, tolerance, updateInterval) {}

  PyCoordinateDescent(double stepSize,
                     size_t maxIterations,
                     double tolerance,
                     size_t updateInterval,
                     const DescentPolicy& descentPolicy)
      : optimizer(stepSize, maxIterations, tolerance, updateInterval, descentPolicy) {}

  double getStepSize() const { return optimizer.StepSize(); }
  void setStepSize(double stepSize) { optimizer.StepSize() = stepSize; }

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }
  void setMaxIterations(size_t maxIterations) { optimizer.MaxIterations() = maxIterations; }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  size_t getUpdateInterval() const { return optimizer.UpdateInterval(); }
  void setUpdateInterval(size_t updateInterval) { optimizer.UpdateInterval() = updateInterval; }

  template<typename FunctionType>
  arma::mat Optimize(FunctionType& function, const arma::mat& initialPoint)
  {
    arma::mat coordinates = initialPoint;
    optimizer.Optimize(function, coordinates);
    return coordinates;
  }

private:
  ens::CD<DescentPolicy> optimizer;
};

// Specific coordinate descent variants
using PyCyclicDescent = PyCoordinateDescent<ens::CyclicDescent>;
using PyRandomDescent = PyCoordinateDescent<ens::RandomDescent>;
using PyGreedyDescent = PyCoordinateDescent<ens::GreedyDescent>;

// Wrapper for Simulated Annealing optimizer
template<typename CoolingSchedule = ens::ExponentialSchedule>
class PySimulatedAnnealing
{
public:
  PySimulatedAnnealing(size_t maxIterations = 1000000,
                      double initT = 10000.0,
                      size_t initMoves = 1000,
                      size_t moveCtrlSweep = 100,
                      double tolerance = 1e-5,
                      size_t maxToleranceSweep = 3,
                      double maxMoveCoef = 20,
                      double initMoveCoef = 0.3,
                      double gain = 0.3)
      : optimizer(CoolingSchedule(), maxIterations, initT, initMoves, 
                 moveCtrlSweep, tolerance, maxToleranceSweep, 
                 maxMoveCoef, initMoveCoef, gain) {}

  PySimulatedAnnealing(const CoolingSchedule& coolingSchedule,
                      size_t maxIterations = 1000000,
                      double initT = 10000.0,
                      size_t initMoves = 1000,
                      size_t moveCtrlSweep = 100,
                      double tolerance = 1e-5,
                      size_t maxToleranceSweep = 3,
                      double maxMoveCoef = 20,
                      double initMoveCoef = 0.3,
                      double gain = 0.3)
      : optimizer(coolingSchedule, maxIterations, initT, initMoves,
                 moveCtrlSweep, tolerance, maxToleranceSweep,
                 maxMoveCoef, initMoveCoef, gain) {}

  size_t getMaxIterations() const { return optimizer.MaxIterations(); }
  void setMaxIterations(size_t maxIterations) { optimizer.MaxIterations() = maxIterations; }

  double getTemperature() const { return optimizer.Temperature(); }
  void setTemperature(double temperature) { optimizer.Temperature() = temperature; }

  size_t getInitMoves() const { return optimizer.InitMoves(); }
  void setInitMoves(size_t initMoves) { optimizer.InitMoves() = initMoves; }

  size_t getMoveCtrlSweep() const { return optimizer.MoveCtrlSweep(); }
  void setMoveCtrlSweep(size_t moveCtrlSweep) { optimizer.MoveCtrlSweep() = moveCtrlSweep; }

  double getTolerance() const { return optimizer.Tolerance(); }
  void setTolerance(double tolerance) { optimizer.Tolerance() = tolerance; }

  size_t getMaxToleranceSweep() const { return optimizer.MaxToleranceSweep(); }
  void setMaxToleranceSweep(size_t maxToleranceSweep) { optimizer.MaxToleranceSweep() = maxToleranceSweep; }

  double getGain() const { return optimizer.Gain(); }
  void setGain(double gain) { optimizer.Gain() = gain; }

  template<typename FunctionType>
  arma::mat Optimize(FunctionType& function, const arma::mat& initialPoint)
  {
    arma::mat coordinates = initialPoint;
    optimizer.Optimize(function, coordinates);
    return coordinates;
  }

private:
  ens::SA<CoolingSchedule> optimizer;
};

// Default simulated annealing with exponential cooling
using PySimulatedAnnealingDefault = PySimulatedAnnealing<ens::ExponentialSchedule>;