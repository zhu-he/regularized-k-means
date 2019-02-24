#ifndef LASSO_K_MEANS_H_
#define LASSO_K_MEANS_H_

#include <random>
#include <vector>

#include "k_means.h"

class LassoKMeans : public KMeans {
 public:
  LassoKMeans(const std::vector<std::vector<double>>& data, int k,
              InitMethod init_method = KMeans::kForgy,
              unsigned int seed = std::random_device{}());
  double Solve(double lambda);
};

#endif  // LASSO_K_MEANS_H_
