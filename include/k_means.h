#ifndef K_MEANS_H_
#define K_MEANS_H_

#include <random>
#include <vector>

class KMeans {
 public:
  enum InitMethod { kForgy, kRandomPartition };
  KMeans(const std::vector<std::vector<double>>& data, int k,
         InitMethod init_method, unsigned int seed);
  const std::vector<std::vector<double>>& cluster_centers() const;
  const std::vector<int>& assignments() const;
  double GetSumSquaredError() const;

 protected:
  void Init();
  double CalDistance(const std::vector<double>& data1,
                     const std::vector<double>& data2) const;
  void UpdateClusterCenter();
  void InitWithRandomCenter();
  void InitWithRandomAssignment();
  const int n_;
  const int s_;
  const int k_;
  const std::vector<std::vector<double>> data_;
  InitMethod init_method_;
  const unsigned int seed_;
  std::vector<std::vector<double>> cluster_centers_;
  std::vector<int> assignments_;
  std::default_random_engine el_;
};

#endif  // K_MEANS_H_
