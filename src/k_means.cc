#include "k_means.h"

KMeans::KMeans(const std::vector<std::vector<double>>& data, int k,
               InitMethod init_method, unsigned int seed)
    : data_(data),
      n_(static_cast<int>(data.size())),
      s_(static_cast<int>(data.front().size())),
      k_(k),
      init_method_(init_method),
      el_(seed),
      seed_(seed) {}

const std::vector<std::vector<double>>& KMeans::cluster_centers() const {
  return this->cluster_centers_;
}

const std::vector<int>& KMeans::assignments() const {
  return this->assignments_;
}

double KMeans::CalDistance(const std::vector<double>& data1,
                           const std::vector<double>& data2) const {
  double result = 0.0;
  for (int i = 0; i < s_; ++i) {
    result += (data1[i] - data2[i]) * (data1[i] - data2[i]);
  }
  return result;
}

void KMeans::Init() {
  InitWithRandomAssignment();
  switch (init_method_) {
    case kForgy:
      InitWithRandomCenter();
      break;
    case kRandomPartition:
      UpdateClusterCenter();
      break;
  }
}

void KMeans::UpdateClusterCenter() {
  cluster_centers_ =
      std::vector<std::vector<double>>(k_, std::vector<double>(s_));
  std::vector<int> cluster_size(k_, 0);
  for (int i = 0; i < n_; ++i) {
    ++cluster_size[assignments_[i]];
    for (int j = 0; j < s_; ++j) {
      cluster_centers_[assignments_[i]][j] += data_[i][j];
    }
  }
  for (int i = 0; i < k_; ++i) {
    if (cluster_size[i] > 0) {
      for (auto& coordinate : cluster_centers_[i]) {
        coordinate /= cluster_size[i];
      }
    } else {
      cluster_centers_[i] =
          data_[std::uniform_int_distribution<int>(0, n_ - 1)(el_)];
    }
  }
}

double KMeans::GetSumSquaredError() const {
  double sum = 0;
  for (int i = 0; i < n_; ++i) {
    sum += CalDistance(data_[i], cluster_centers_[assignments_[i]]);
  }
  return sum;
}

void KMeans::InitWithRandomCenter() {
  cluster_centers_.resize(k_);
  std::vector<int> indices(n_);
  for (int i = 0; i < n_; ++i) {
    indices[i] = i;
  }
  for (int i = 0; i < k_; ++i) {
    int pos = std::uniform_int_distribution<int>(i, n_ - 1)(el_);
    std::swap(indices[i], indices[pos]);
  }
  for (int i = 0; i < k_; ++i) {
    cluster_centers_[i] = data_[indices[i]];
  }
}

void KMeans::InitWithRandomAssignment() {
  assignments_.resize(n_);
  std::uniform_int_distribution<int> dist(0, k_ - 1);
  for (int i = 0; i < n_; ++i) {
    assignments_[i] = dist(el_);
  }
}
