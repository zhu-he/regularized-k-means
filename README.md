# Balanced Clustering: A Uniform Model and Fast Algorithm

### IJCAI-19 Paper ID: 6035

## Build

### Linux

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Then the executable file is under `./build/`

### Windows (Visual Studio)

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Then the executable file is under `.\build\Release\`

## Run

```
./regularized-k-means [type] [file] [k] {OPTIONS}

  OPTIONS:

      -h, --help                        Display this help menu
      type                              Algorithm type
                                        - 'hard': clustering under the strict
                                                  balance constraint
                                        - 'soft': with lambda*x^2 regularization
                                        - 'lasso': our implementation of the
                                                   lasso k-means algorithm
                                                   proposed by Li et al. [2018].
      file                              Data file
      k                                 Number of clusters
      -i[init], --init=[init]           Init method
                                        - 'forgy': Default. The Forgy method
                                                   randomly chooses k data
                                                   points from the dataset
                                                   and uses these as the initial
                                                   means.
                                        - 'rp': The Random Partition method
                                                first randomly assigns a cluster
                                                to each data point and then
                                                proceeds to the update step,
                                                thus computing the initial mean
                                                to be the centroid of the
                                                cluster's randomly assigned
                                                points.
      -n, --no-warm-start               Turn off warm start
      -t[threads], --threads=[threads]  Number of threads for parallel computing
                                        of the cost matrix, or '-1' for auto
                                        detecting the hardware concurrency.
                                        Default is 1.
      -s[seed], --seed=[seed]           Random seed
      -l[lambda], --lambda=[lambda]     Lambda (required when type equals 'soft'
                                        or 'lasso')
      -r[runs], --runs=[runs]           Number of runs
      -a[file], --assignment=[file]     Place the result of assignments into
                                        [file].csv. If multiple runs is enabled,
                                        [file]-<i>.csv is placed in i-th run.
      -c[file], --cluster=[file]        Place the result of cluster centers into
                                        [file].csv. If multiple runs is enabled,
                                        [file]-<i>.csv is placed in i-th run.
      -o[file], --output=[file]         Append the summary of results into
                                        [file] in format 'type,file,k,init,
                                        warm_start,threads,seed,lambda,
                                        sum_of_squares,used_time'
      "--" can be used to terminate flag options and force all following
      arguments to be treated as positional options

    [Li et al., 2018] Zhihui Li, Feiping Nie, Xiaojun Chang, Zhigang Ma, and Yi
    Yang. Balanced clustering via exclusive lasso: A pragmatic approach. In
    Thirty-Second AAAI Conference on Artificial Intelligence, 2018.
```

## Examples

```shell
$ ./regularized-k-means hard data/iris.csv 3 -a assignments -c clusters -o summary.txt -r3

Sum of Squares: 81.3672
Used Time: 0.000208666
Sum of Squares: 81.3672
Used Time: 0.000153128
Sum of Squares: 81.3672
Used Time: 0.000161978

$ ./regularized-k-means hard data/vowel.csv 11 -s42

Sum of Squares: 1.95543e+07
Used Time: 0.0179554

$ ./regularized-k-means hard data/vowel.csv 11 -s42 -n

Sum of Squares: 1.95543e+07
Used Time: 0.253261

$ ./regularized-k-means soft data/user_knowledge.csv 4 -s42 -l0.005

Sum of Squares: 72.1259
Used Time: 0.00677741

$ ./regularized-k-means hard data/mnist_train.csv 10 -s42

Sum of Squares: 1.5356e+11
Used Time: 23.2523

$ ./regularized-k-means hard data/mnist_train.csv 10 -s42 -t-1

Sum of Squares: 1.5356e+11
Used Time: 8.43786
```

## Custom regularizers

The custom regularizers need to be manually implemented.
Use a anonymous function in C++11:

```cpp
RegularizedKMeans rkm(data, k);
double result = rkm.Solve([lambda](int h, int x) -> double {
    // f_h(x) function is defined here
    return lambda * x * x;
});
```

or use a old-school function definition

```cpp
double f(int h, int x) {
    // f_h(x) function is defined here
    return lambda * x * x;
}
// ...
RegularizedKMeans rkm(data, k);
double result = rkm.Solve(f);
```
