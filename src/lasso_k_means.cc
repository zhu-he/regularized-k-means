#include "lasso_k_means.h"

int Square(int x) { return x * x; }

LassoKMeans::LassoKMeans(const std::vector<std::vector<double>>& data, int k,
                         InitMethod init_method, unsigned int seed)
    : KMeans(data, k, init_method, seed) {}

double LassoKMeans::Solve(double lambda) {
  Init();
  while (true) {
    std::vector<int> cluster_size(k_, 0);
    for (int i = 0; i < n_; ++i) {
      ++cluster_size[assignments_[i]];
    }
    bool changed = false;
    for (int i = 0; i < n_; ++i) {
      double best_value = 0.0;
      int best_cluster = assignments_[i];
      double base = -CalDistance(data_[i], cluster_centers_[assignments_[i]]) -
                    lambda * Square(cluster_size[assignments_[i]]) +
                    lambda * Square(cluster_size[assignments_[i]] - 1);
      for (int j = 0; j < k_; ++j) {
        if (j == assignments_[i]) {
          continue;
        }
        double delta = base + CalDistance(data_[i], cluster_centers_[j]) +
                       lambda * Square(cluster_size[j] + 1) -
                       lambda * Square(cluster_size[j]);
        if (delta < best_value) {
          best_value = delta;
          best_cluster = j;
        }
      }
      if (best_cluster != assignments_[i]) {
        --cluster_size[assignments_[i]];
        ++cluster_size[best_cluster];
        assignments_[i] = best_cluster;
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
    UpdateClusterCenter();
  }
  return GetSumSquaredError();
}
