cmake_minimum_required(VERSION 3.3)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)
include_directories(include)
include_directories(third_party)

add_executable(regularized-k-means
               src/k_means.cc
               src/lasso_k_means.cc
               src/main.cc
               src/network_simplex.cc
               src/regularized_k_means.cc)

target_link_libraries(regularized-k-means ${CMAKE_THREAD_LIBS_INIT})
