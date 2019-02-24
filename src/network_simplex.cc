#include "network_simplex.h"

void NetworkSimplex::BuildHard(const std::vector<std::vector<double>>& costs,
                               int k, int lower_bound, int upper_bound) {
  const std::vector<int>& sum_flow = BuildBasic(costs, 1);
  for (int i = 0; i < k_; ++i) {
    auto& edge = edge_list_[n_ * k_ + i];
    edge.from = n_ + 1 + i;
    edge.to = 0;
    edge.cap = upper_bound - lower_bound;
    edge.flow = sum_flow[i] - lower_bound;
    edge.cost = 0.0;
    edge.in_tree = true;
  }
  BuildTree();
}

void NetworkSimplex::Build(const std::vector<std::vector<double>>& costs,
                           const std::function<double(int, int)>& f) {
  const std::vector<int>& sum_flow =
      BuildBasic(costs, static_cast<int>(costs.size()));
  for (int i = 0; i < k_; ++i) {
    for (int j = 0; j < n_; ++j) {
      auto& edge = edge_list_[n_ * k_ + i * n_ + j];
      edge.from = n_ + 1 + i;
      edge.to = 0;
      edge.flow = sum_flow[i] >= j + 1;
      edge.in_tree = j == 0;
      edge.cap = 1;
      edge.cost = f(i, j + 1) - f(i, j);
    }
  }
  BuildTree();
}

std::vector<int> NetworkSimplex::BuildBasic(
    const std::vector<std::vector<double>>& costs, int extra_edge_num_) {
  n_ = static_cast<int>(costs.size());
  k_ = static_cast<int>(costs.front().size());
  std::vector<int> sum_flow(k_, 0);
  for (int i = 0; i < n_; ++i) {
    ++sum_flow[i % k_];
  }
  int vertex_num = n_ + k_ + 1;
  int edge_num = n_ * k_ + k_ * extra_edge_num_;
  parent_.resize(vertex_num);
  parent_edge_index_.resize(vertex_num);
  parent_direction_.resize(vertex_num);
  vis_.resize(vertex_num);
  potential_.resize(vertex_num);
  potential_tag_.resize(vertex_num, -1);
  edge_list_.resize(edge_num);
  potential_tag_[0] = tag_ = 0;
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < k_; ++j) {
      auto& edge = edge_list_[i * k_ + j];
      edge.from = i + 1;
      edge.to = n_ + 1 + j;
      edge.cap = 1;
      edge.flow = 0;
      edge.cost = costs[i][j];
      edge.in_tree = false;
    }
  }
  for (int i = 0; i < n_; ++i) {
    int j = i % k_;
    edge_list_[i * k_ + j].flow = 1;
    edge_list_[i * k_ + j].in_tree = true;
  }
  return sum_flow;
}

void NetworkSimplex::BuildTree() {
  min_cost_ = 0;
  for (int i = 0; i < static_cast<int>(edge_list_.size()); ++i) {
    min_cost_ += edge_list_[i].flow * edge_list_[i].cost;
    if (edge_list_[i].in_tree) {
      parent_[edge_list_[i].from] = edge_list_[i].to;
      parent_edge_index_[edge_list_[i].from] = i;
      parent_direction_[edge_list_[i].from] = 1;
    }
  }
}

void NetworkSimplex::Simplex() {
  int num_edges = static_cast<int>(edge_list_.size());
  for (int edge_index = 0, scaned = 0; scaned < num_edges;
       ++edge_index, ++scaned) {
    if (edge_index == num_edges) {
      edge_index = 0;
    }
    Edge& edge = edge_list_[edge_index];
    if (edge.in_tree || edge.cap == 0) {
      continue;
    }
    double potential_from = GetPotential(edge.from);
    double potential_to = GetPotential(edge.to);
    int direction = edge.flow == 0 ? 1 : -1;
    double delta = (potential_to - potential_from + edge.cost) * direction;
    if (delta < -kEps) {
      Pivot(edge_index, direction, delta);
      scaned = 0;
    }
  }
}

void NetworkSimplex::UpdateCosts(
    const std::vector<std::vector<double>>& costs) {
  int n_ = static_cast<int>(costs.size());
  for (auto& edge : edge_list_) {
    if (1 <= edge.from && edge.from <= n_) {
      if (edge.flow == 1) {
        min_cost_ += costs[edge.from - 1][edge.to - n_ - 1] - edge.cost;
      }
      edge.cost = costs[edge.from - 1][edge.to - n_ - 1];
    }
  }
  potential_tag_[0] = ++tag_;
}

void NetworkSimplex::GetAssignments(std::vector<int>* assignments) const {
  assignments->resize(n_);
  for (const auto& edge : edge_list_) {
    if (edge.flow == 1 && 1 <= edge.from && edge.from <= n_) {
      (*assignments)[edge.from - 1] = edge.to - n_ - 1;
    }
  }
}

double NetworkSimplex::min_cost() const { return min_cost_; }

void NetworkSimplex::Pivot(int edge_index, int direction, double delta) {
  Edge& edge = edge_list_[edge_index];
  int min_res_cap = edge.cap;
  int min_res_cap_edge_index = -1;
  int min_res_direction = 0;
  int lca = FindLca(edge.from, edge.to);
  int current_node = edge.from;
  while (current_node != lca) {
    int res_cap = GetParentResCap(current_node, -direction);
    if (res_cap < min_res_cap) {
      min_res_cap = res_cap;
      min_res_cap_edge_index = current_node;
      min_res_direction = 1;
    }
    current_node = parent_[current_node];
  }
  current_node = edge.to;
  while (current_node != lca) {
    int res_cap = GetParentResCap(current_node, direction);
    if (res_cap < min_res_cap) {
      min_res_cap = res_cap;
      min_res_cap_edge_index = current_node;
      min_res_direction = -1;
    }
    current_node = parent_[current_node];
  }
  if (min_res_cap > 0) {
    min_cost_ += min_res_cap * delta;
    edge.flow += direction * min_res_cap;
    current_node = edge.from;
    while (current_node != lca) {
      ApplyParentFlow(current_node, -direction, min_res_cap);
      current_node = parent_[current_node];
    }
    current_node = edge.to;
    while (current_node != lca) {
      ApplyParentFlow(current_node, direction, min_res_cap);
      current_node = parent_[current_node];
    }
  }
  if (min_res_direction != 0) {
    potential_tag_[0] = ++tag_;
    edge_list_[parent_edge_index_[min_res_cap_edge_index]].in_tree = false;
    edge.in_tree = true;
    current_node = min_res_direction == 1 ? edge.from : edge.to;
    ChangeDirection(current_node, min_res_cap_edge_index);
    parent_edge_index_[current_node] = edge_index;
    parent_[current_node] = edge.from ^ edge.to ^ current_node;
    parent_direction_[current_node] = min_res_direction;
  }
}

double NetworkSimplex::GetPotential(int u) {
  if (potential_tag_[u] != tag_) {
    potential_[u] =
        GetPotential(parent_[u]) +
        parent_direction_[u] * edge_list_[parent_edge_index_[u]].cost;
  }
  potential_tag_[u] = tag_;
  return potential_[u];
}

int NetworkSimplex::GetParentResCap(int u, int direction) {
  if (direction * parent_direction_[u] > 0) {
    return edge_list_[parent_edge_index_[u]].cap -
           edge_list_[parent_edge_index_[u]].flow;
  } else {
    return edge_list_[parent_edge_index_[u]].flow;
  }
}

void NetworkSimplex::ApplyParentFlow(int u, int direction, int flow) {
  edge_list_[parent_edge_index_[u]].flow +=
      direction * parent_direction_[u] * flow;
}

void NetworkSimplex::ChangeDirection(int u, int end) {
  if (u != end) {
    ChangeDirection(parent_[u], end);
    parent_[parent_[u]] = u;
    parent_edge_index_[parent_[u]] = parent_edge_index_[u];
    parent_direction_[parent_[u]] = -parent_direction_[u];
  }
}

int NetworkSimplex::FindLca(int u, int v) {
  int t = u;
  while (t) {
    vis_[t] = true;
    t = parent_[t];
  }
  while (v && !vis_[v]) {
    v = parent_[v];
  }
  while (u) {
    vis_[u] = false;
    u = parent_[u];
  }
  return v;
}
