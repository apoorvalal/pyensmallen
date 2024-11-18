#pragma once
#include <armadillo>
#include <ensmallen.hpp>

// Solver for constraints where weights must be positive and sum to 1
class ConstrSimplexSolver
{
public:
  ConstrSimplexSolver() {}

  template <typename VecType>
  void Project(VecType &coords)
  {
    // First ensure non-negative
    for (size_t i = 0; i < coords.n_elem; ++i)
    {
      if (coords[i] < 0)
        coords[i] = 0;
    }

    // Then normalize to sum to 1
    double sum = arma::accu(coords);
    if (sum > 0)
      coords /= sum;
    else
      coords.fill(1.0 / coords.n_elem); // Uniform distribution if all zeros
  }

  template <typename VecType>
  void GetVertex(const VecType &gradient, VecType &vertex)
  {
    // Initialize vertex to zero vector
    vertex.zeros(gradient.n_elem);

    // Find index of minimum gradient component
    size_t minIndex = 0;
    double minValue = gradient[0];

    for (size_t i = 1; i < gradient.n_elem; ++i)
    {
      if (gradient[i] < minValue)
      {
        minValue = gradient[i];
        minIndex = i;
      }
    }

    // Put all weight on that coordinate
    vertex[minIndex] = 1.0;
  }

  template <typename VecType>
  bool IsFeasible(const VecType &coords)
  {
    // Check if all coordinates are non-negative
    for (size_t i = 0; i < coords.n_elem; ++i)
    {
      if (coords[i] < 0)
        return false;
    }

    // Check if sum is approximately 1
    double sum = arma::accu(coords);
    return std::abs(sum - 1.0) < 1e-10;
  }
};
