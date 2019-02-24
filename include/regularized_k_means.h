#ifndef REGULARIZED_K_MEANS_H_
#define REGULARIZED_K_MEANS_H_

#include <functional>
#include <random>
#include <vector>

#include "k_means.h"
#include "network_simplex.h"

class RegularizedKMeans : public KMeans {
 public:
  RegularizedKMeans(const std::vector<std::vector<double>>& data, int k,
                    InitMethod init_method = KMeans::kForgy,
                    bool warm_start = true, int n_jobs = 1,
                    unsigned int seed = std::random_device{}());
  double SolveHard();
  double SolveHard(int lower_bound, int upper_bound);
  double Solve(const std::function<double(int, int)>& f);

 protected:
  double Solve(std::function<NetworkSimplex()> builder);
  void UpdateCostMatrix();
  const bool warm_start_;
  const int n_jobs_;
  std::vector<std::vector<double>> costs_;
};

#endif  // REGULARIZED_K_MEANS_H_
