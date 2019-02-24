#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <args.hxx>

#include "lasso_k_means.h"
#include "regularized_k_means.h"

std::vector<std::vector<double>> ReadData(const std::string& file_name) {
  std::vector<std::vector<double>> data;
  std::ifstream file(file_name);
  std::string line;
  while (std::getline(file, line)) {
    for (auto& c : line) {
      if (c == ',') {
        c = ' ';
      }
    }
    std::stringstream ss(line);
    std::vector<double> line_data;
    double value;
    while (ss >> value) {
      line_data.emplace_back(value);
    }
    data.emplace_back(line_data);
  }
  return data;
}

template <class T>
std::string GetKeyByValue(const std::unordered_map<std::string, T>& map,
                          T value) {
  for (const auto& kv : map) {
    if (kv.second == value) {
      return kv.first;
    }
  }
  return std::string();
}

void WriteAssignments(const std::string& file_name,
                      const std::vector<int>& assignments) {
  std::ofstream file(file_name);
  for (auto assignment : assignments) {
    file << assignment << '\n';
  }
}

void WriteClusterCenters(
    const std::string& file_name,
    const std::vector<std::vector<double>>& cluster_centers) {
  std::ofstream file(file_name);
  for (const auto& cluster_center : cluster_centers) {
    bool first = true;
    for (int i = 0; i < static_cast<int>(cluster_center.size()); ++i) {
      if (!first) {
        file << ',';
      }
      first = false;
      file << cluster_center[i];
    }
    file << '\n';
  }
}

int main(int argc, char* argv[]) {
  enum AlgorithmType { kHard, kSoft, kLasso };
  std::unordered_map<std::string, AlgorithmType> type_map{
      {"hard", AlgorithmType::kHard},
      {"soft", AlgorithmType::kSoft},
      {"lasso", AlgorithmType::kLasso}};
  std::unordered_map<std::string, RegularizedKMeans::InitMethod>
      init_method_map{{"forgy", RegularizedKMeans::kForgy},
                      {"rp", RegularizedKMeans::kRandomPartition}};

  args::ArgumentParser parser(
      "Balanced Clustering: A Uniform Model and Fast Algorithm\n"
      "IJCAI-19 Paper ID: 6035",
      "[Li et al., 2018] Zhihui Li, Feiping Nie, Xiaojun Chang, Zhigang Ma, "
      "and Yi Yang. Balanced clustering via exclusive lasso: A pragmatic "
      "approach. In Thirty-Second AAAI Conference on Artificial Intelligence, "
      "2018.\n");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::Group required(parser, "", args::Group::Validators::All);
  args::MapPositional<std::string, AlgorithmType> type(
      required, "type",
      "Algorithm type\n"
      "- 'hard': clustering under the strict\n"
      "          balance constraint\n"
      "- 'soft': with lambda*x^2 regularization\n"
      "- 'lasso': our implementation of the\n"
      "           lasso k-means algorithm\n"
      "           proposed by Li et al. [2018].",
      type_map);
  args::Positional<std::string> file(required, "file", "Data file");
  args::Positional<int> k(required, "k", "Number of clusters");
  args::MapFlag<std::string, RegularizedKMeans::InitMethod> init_method(
      parser, "init",
      "Init method\n"
      "- 'forgy': Default. The Forgy method\n"
      "           randomly chooses k data\n"
      "           points from the dataset\n"
      "           and uses these as the initial\n"
      "           means.\n"
      "- 'rp': The Random Partition method\n"
      "        first randomly assigns a cluster\n"
      "        to each data point and then\n"
      "        proceeds to the update step,\n"
      "        thus computing the initial mean\n"
      "        to be the centroid of the\n"
      "        cluster's randomly assigned\n"
      "        points.\n",
      {'i', "init"}, init_method_map, RegularizedKMeans::InitMethod::kForgy);
  args::Flag no_warm_start(parser, "no-warm-start", "Turn off warm start",
                           {'n', "no-warm-start"});
  args::ValueFlag<int> threads(
      parser, "threads",
      "Number of threads for parallel computing of the cost matrix, or '-1' "
      "for auto detecting the hardware concurrency. Default is 1.",
      {'t', "threads"}, 1);
  args::ValueFlag<unsigned int> seed(parser, "seed", "Random seed",
                                     {'s', "seed"}, std::random_device{}());
  args::ValueFlag<double> lambda(
      parser, "lambda", "Lambda (required when type equals 'soft' or 'lasso')",
      {'l', "lambda"}, 0);
  args::ValueFlag<int> runs(parser, "runs", "Number of runs", {'r', "runs"}, 1);
  args::ValueFlag<std::string> assignment_file(
      parser, "file",
      "Place the result of assignments into [file].csv. If multiple runs is "
      "enabled, [file]-<i>.csv is placed in i-th run.",
      {'a', "assignment"});
  args::ValueFlag<std::string> cluster_center_file(
      parser, "file",
      "Place the result of cluster centers into [file].csv. If multiple runs "
      "is enabled, [file]-<i>.csv is placed in i-th run.",
      {'c', "cluster"});
  args::ValueFlag<std::string> summary_file(
      parser, "file",
      "Append the summary of results into\n"
      "[file] in format 'type,file,k,init,\n"
      "warm_start,threads,seed,lambda,\n"
      "sum_of_squares,used_time'",
      {'o', "output"});
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  auto data = ReadData(args::get(file));
  for (int run = 1; run <= args::get(runs); ++run) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result;
    KMeans* k_means;
    if (args::get(type) == AlgorithmType::kLasso) {
      auto lkm = new LassoKMeans(data, args::get(k), args::get(init_method),
                                 args::get(seed) + run - 1);
      result = lkm->Solve(args::get(lambda));
      k_means = lkm;
    } else {
      auto* rkm = new RegularizedKMeans(
          data, args::get(k), args::get(init_method), !no_warm_start,
          args::get(threads), args::get(seed) + run - 1);
      if (args::get(type) == AlgorithmType::kHard) {
        result = rkm->SolveHard();
      } else {
        double lambda_value = args::get(lambda);
        result = rkm->Solve([lambda_value](int h, int x) -> double {
          return lambda_value * x * x;
        });
      }
      k_means = rkm;
    }
    std::string run_suffix =
        args::get(runs) == 1 ? "" : "-" + std::to_string(run);
    double used_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start_time)
            .count();
    if (!args::get(assignment_file).empty()) {
      WriteAssignments(args::get(assignment_file) + run_suffix + ".csv",
                       k_means->assignments());
    }
    if (!args::get(cluster_center_file).empty()) {
      WriteClusterCenters(args::get(cluster_center_file) + run_suffix + ".csv",
                          k_means->cluster_centers());
    }
    if (!args::get(summary_file).empty()) {
      std::fstream out;
      out.open(args::get(summary_file), std::fstream::app);
      out << GetKeyByValue(type_map, args::get(type)) << ',' << args::get(file)
          << ',' << args::get(k) << ','
          << GetKeyByValue(init_method_map, args::get(init_method)) << ','
          << std::boolalpha << !no_warm_start << ',' << args::get(threads)
          << ',' << args::get(seed) + run - 1 << ',' << args::get(lambda) << ','
          << result << ',' << used_time << std::endl;
    }
    std::cerr << "Sum of Squares: " << result << std::endl;
    std::cerr << "Used Time: " << used_time << std::endl;
    delete k_means;
  }
  return 0;
}
