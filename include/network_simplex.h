#ifndef NETWORK_SIMPLEX_H_
#define NETWORK_SIMPLEX_H_

#include <functional>
#include <vector>

class NetworkSimplex {
 public:
  void BuildHard(const std::vector<std::vector<double>>& costs, int k,
                 int lower_bound, int upper_bound);
  void Build(const std::vector<std::vector<double>>& costs,
             const std::function<double(int, int)>& f);
  void Simplex();
  void UpdateCosts(const std::vector<std::vector<double>>& costs);
  void GetAssignments(std::vector<int>* assignments) const;
  double min_cost() const;

 private:
  std::vector<int> BuildBasic(const std::vector<std::vector<double>>& costs,
                              int extra_edge_num_);
  void BuildTree();
  void Pivot(int edge_index, int direction, double delta);
  double GetPotential(int u);
  int GetParentResCap(int u, int direction);
  void ApplyParentFlow(int u, int direction, int flow);
  void ChangeDirection(int u, int end);
  int FindLca(int u, int v);
  struct Edge {
    int from;
    int to;
    int cap;
    int flow;
    double cost;
    bool in_tree;
  };
  std::vector<int> parent_;
  std::vector<int> parent_edge_index_;
  std::vector<int> parent_direction_;
  std::vector<bool> vis_;
  std::vector<Edge> edge_list_;
  std::vector<double> potential_;
  std::vector<int> potential_tag_;
  int tag_;
  double min_cost_;
  int n_;
  int k_;
  static constexpr double kEps = 1e-6;
};

#endif  // NETWORK_SIMPLEX_H_
