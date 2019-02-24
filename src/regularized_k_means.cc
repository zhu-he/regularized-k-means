#include "regularized_k_means.h"

#include <algorithm>
#include <thread>

RegularizedKMeans::RegularizedKMeans(
    const std::vector<std::vector<double>>& data, int k, InitMethod init_method,
    bool warm_start, int n_jobs, unsigned int seed)
    : KMeans(data, k, init_method, seed),
      warm_start_(warm_start),
      n_jobs_(n_jobs == -1 ? std::thread::hardware_concurrency() : n_jobs),
      costs_(static_cast<int>(data.size()), std::vector<double>(k)) {}

double RegularizedKMeans::SolveHard() {
  return SolveHard(n_ / k_, (n_ + k_ - 1) / k_);
}

double RegularizedKMeans::SolveHard(int lower_bound, int upper_bound) {
  return Solve([this, lower_bound, upper_bound]() -> NetworkSimplex {
    NetworkSimplex ns = NetworkSimplex();
    ns.BuildHard(this->costs_, this->k_, lower_bound, upper_bound);
    return ns;
  });
}

double RegularizedKMeans::Solve(const std::function<double(int, int)>& f) {
  return Solve([this, &f]() -> NetworkSimplex {
    NetworkSimplex ns = NetworkSimplex();
    ns.Build(this->costs_, f);
    return ns;
  });
}

double RegularizedKMeans::Solve(std::function<NetworkSimplex()> builder) {
  Init();
  UpdateCostMatrix();
  std::vector<int> old_assignments;
  NetworkSimplex ns_solver = builder();
  ns_solver.Simplex();
  ns_solver.GetAssignments(&assignments_);
  do {
    old_assignments = assignments_;
    UpdateClusterCenter();
    UpdateCostMatrix();
    if (warm_start_) {
      ns_solver.UpdateCosts(costs_);
    } else {
      ns_solver = builder();
    }
    ns_solver.Simplex();
    ns_solver.GetAssignments(&assignments_);
  } while (old_assignments != assignments_);
  return GetSumSquaredError();
}

void RegularizedKMeans::UpdateCostMatrix() {
  if (n_jobs_ <= 1) {
    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < k_; ++j) {
        costs_[i][j] = CalDistance(data_[i], cluster_centers_[j]);
      }
    }
  } else {
    std::vector<std::thread> threads(n_jobs_);
    for (int t = 0; t < n_jobs_; ++t) {
      threads[t] = std::thread(std::bind(
          [this](int thread_idx) {
            for (int idx = thread_idx; idx < n_ * k_; idx += n_jobs_) {
              int i = idx / k_;
              int j = idx % k_;
              costs_[i][j] = CalDistance(data_[i], cluster_centers_[j]);
            }
          },
          t));
    }
    std::for_each(threads.begin(), threads.end(),
                  [](std::thread& x) { x.join(); });
  }
}
