# mlmc-cpp

`mlmc-cpp` is an implementation of MLMC(Multilevel Monte Carlo method) to the calclation of EVPPI(Expected Value of Partial Perfect information).
MLMC enables us to calc EVPPI faster than NMC(Nested Monte Carlo method).

see https://arxiv.org/abs/1708.05531 .

# Usage

## if you run existing models

Run commands bellow.
The output is written both on your console (standard output) and in `output.txt` which is ignored by git.

```
$ cd any_model
$ make
$ ./evppi
```

## if you run a new model

### 1. create a new directory and new files

```
$ mkdir new_model
$ cd new_model
$ touch new_model.cpp Makefile
```

### 2. edit cpp file

You need to define 1 struct and 4 functions.
Here is an example of a very simple model (see `./test`).
If you need Matrix structure or other complicated function, see `./matrix.hpp` or `./util.hpp`.

``` cpp
struct ModelInfo {
  // variables of the model
  // example
  double x, y;
}

void sampling_init(EvppiInfo *info) {
  // define how many model functions are 
  info->model_num = ...
  
  // allocate memory for ModelInfo
  info->model_info = new ModelInfo;
  
  // allocate memory for outputs
  info->val.resize(info->model_num);
}

void pre_sampling(ModelInfo *model) {
  // generate outer variables of which you are calclating EVPPI
  model->x = ...
}

void post_sampling(ModelInfo *model) {
  // generate inner variables
  model->y = ...
}

void f(EvppiInfo *info) {
  // after you generate all the variables, calc the model functions
  // store the outputs in `info->val[i]`
  info->val[0] = ...
  info->val[1] = ...
}

int main() {
  MlmcInfo *info = mlmc_init(1, 2, 30, 1.0, 0.25);
  
  // calc EVPI (because this program outputs `EVPI - EVPPI` value)
  smc_evpi_calc(info->layer[0].evppi_info, 1000000);
  
  mlmc_test(info, 10, 200000);
  
  // define epsirons you need
  vector <double> eps = {...};
  mlmc_test_eval_eps(info, eps);
}
```

### 3. edit Makefile

```
PROGRAM = evppi
SRCS = new_model.cpp ../evppi.cpp
OBJS = $(SRCS:.cpp = .o)
CC = g++
CFLAGS = -Wall -Wextra -std=c++11 -o3

$(PROGRAM) : $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o
```
