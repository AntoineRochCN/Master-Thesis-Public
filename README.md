  # Managing latencies in RL using a SAC variant [WIP]

  ## Overview
  This repository supports my master's thesis on latency management in reinforcement learning (RL) using the DADAC variant of the SAC algorithm, based on the [DADAC article](https://openreview.net/forum?id=Y9cVrdYn10).

  The project has two main objectives:
  1.  **Reproduction**: Re-implementing DADAC (Delay Aware SAC) to verify the hypotheses and results presented in the original paper, as the original code was unavailable.
  2.  **Application**: Applying the method to a custom trading environment. While latency management is often studied in robotics, this project tackles the specific latencies found in algorithmic trading.

  ## Installation

  **Requirements**: Python 3.10+ (Recommended).

  > **Note on JAX/CUDA**: `jax 0.6.2` has compatibility issues with `nvidia-cublas-cu12 12.8`, but `torch` requires this version. The installation commands below overwrite the package with a compatible version.

  ```bash
  pip install -r requirements.txt
  pip install --no-deps --force-reinstall nvidia-cublas-cu12>=12.9.1.4
  ```

  ## Quick Start

  1.  **Collect Data**: Scrape and process Binance data.
      ```bash
      cd data/scrapper
      python python_scrapper.py
      cd ..
      python transform_scrapped_data.py -n my_beautiful_data
      ```
  2.  **Train Models**: Run the comparison script.
      ```bash
      cd custom_env
      python algo_comparison_JAX.py
      ```
  3.  **Train Final Agent**: Train a specific DADAC model.
      ```bash
      cd custom_env
      python run_DADAC.py
      ```
  4.  **Backtest**: Run offline or online tests.
      ```bash
      python backtest_run.py
      ```

  ---

  ## Technical Architecture

  ### 1. Algorithms & Verification
  To ensure correctness, the implementation is based on **Stable-Baselines3** and its JAX fork, **SBX**.
  The key points to navigate in the repository are the following:
  - Four algorithms are to be compared, SAC, SAC_VC (SAC + Value Correction), DSAC (Distributional SAC) and DADAC (Delay Aware SAC). They all rely on eponymous files, with a common structure set in the utils_SAC.py file. \
  Their main differences stand in the critic loss computation:
    - DSAC changes the critic network to compute the parameters of a gaussian distribution, instead of a scalar value. It then adds a loss based on the variance of the distribution.
    - Value Correction is a variant of the n-step bootstrapping method (averages the $n$ forward observations critic values), where the weights correspond to the observed latency distribution (observation + action).
    - DADAC is the mix of both methods
  - To simulate the latency, wrappers were introduced in the utils_env.py file. To do so, all the latencies are pre-computed when resetting the environment, and the actions are chosen / applied only when the timestep corresponds to $t + \tau_{\text{observation}}$ (time to compute an action) and $t + \tau_{\text{observation}} + t_{\text{action}}$ (time to apply the action to the environment). \
  Between two timesteps corresponding to the application of an action over the environment, the algorithm assumes that the past action is still beeing applied to the environment.

  ### 2. JAX Implementation Details

  To implement the trading environment, I fully switched to JAX. Thus, the environment, the wrappers and the RL algorithms entirely run on a GPU, which increase the overall speed of the algorithm by roughly x40, without parallelizing the environments. However, it implies more memory consumption since all the buffers were set on the GPU, and the training metrics are only available at the end of a training.\
  This led to some adaptations to the algorithm:
  - One loop of collecting observations makes one critic update. The policy is updated every $n_{\text{policy}}$ times the critic is updated. The metrics are recorded every $n_{\text{record}}$
  times the policy is updated, and the policy is tested every $n_{\text{test}}$ times the environment is recorded. Consequently, one has to be careful when choosing the parametrization. \
  This is for performance improvements, since is simplifies by a lot the computation of the XLA graph, and produce a +37% speed increase, without losing the SAC logic. 
  - JAX struggles to update arrays in-place when on the GPU. It requires fixed-size arrays and doesn't behave very well with stochastic behaviors. Consequently, some parts of the code do more iterations than required (e.g., padding loops), in order to maintain fixed-size items.
  - The policy always computes the action from the observation, even if the timestep doesn't correspond to the latency. This avoids JAX `if` statements (control flow), which slows down the computation more than passing through an MLP (256,256,256) due to XLA graph constraints.

  ### 3. Trading Environment

  The environment is based on **Binance WebSocket** values, specifically the order book sent every 100ms.
  The data used for training was scraped from this source. The other data used was the 1s candlesticks from the same API.
  All functions related to the trading environment reside in the `custom_env` folder.

  The normalization is detailed in the env.py file, as for the environment step behavior. The reward function can be tuned in the reward_function.py file. The buffers also changed a little to be JAX compatible, but kept the same structure (buffer.py).
  The data is almost trained raw:
    - Only the first values of bid and ask are kept. They are normalized within the first value of the episode. The volume corresponds to the sum of the first 5 bid / ask values.
    - The portfolio value begins at 1, so an ending value of 0.95 means a 5% loss.
    - The trading fees and leverage can be adapted.
    - The "oracle" function corresponds to the maximum value a portfolio can reach by the end of an episode, computed within dynamic programming. It is very useful to measure how good the algorithm behaves (the volatility may sometimes not be sufficient to generate profit).
    - Staying neutral refers to the action 0, going long to 1 and going short to 2.
    - Since the action space is discrete ($\{0, 1, 2\}$), the SAC had to be modified to predict discrete actions. This leads to some variations in the output of the critic and policy networks, along with other kind of behavior of the losses / the entropy.
    - **Warmup**: Understanding how trading works is a hard task for a RL algorithm. To deal with that problem, a first warmup step is set overfitting the environment over the best actions found within dynamic programming.

  ---

  ## Detailed Usage Guide

  ### For the DADAC results verification

  To verify DADAC's results (and the comparison with other algorithm), one can run parallel_run.py the following way:
  ```bash
  python parallel_run.py
  ```
  The results will be store in the folder parallel_run/sim_number_... according to the simulation you are doing. \
  The simulation 0 verifies there's no definition bug in the objects. \
  The simulation 1, 2 and 3 correspond to the simulations made to match the experiments made in DADAC. \
  One can create its own simulation parametrization making a new case, but not all hyperparameters are directly available. Moreover, one can define a new latency distribution for the actions or observations in the function utils_env/get_latency_prop. \
  When launching a simulation, it is proposed to use CUDA or not, and the number of threads to use. I would recommend at least 5 threads using CUDA to maximize the GPU utilization (done on a RTX 3080ti laptop), and to speed up the whole testing, one can run another simulation on CPU this time, using 2 or 3 threads. Using too many threads leads to concurrency over RAM and GPU usage, leading to a drop in performance.

  ### Getting the data

  Before training a model in the trading environment, one has to collect data from Binance first. To do so, two scrapers are available at `data/scrapper`, one in C++ and the other in Python. 
  To install the C++ one, it's required to get the Boost lib, whereas the Python one is a bit slower but easier to use.
  For the C++ scrapper:
  ```bash
  cd data/scrapper
  cmake .
  make scrapper_exec
  ./scrapper_exec 
  ```
  For the Python scrapper:
  ```bash
  cd data/scrapper
  python python_scrapper.py
  ```
  Each one will store the results into data/scrapper/scrapper_out.csv. If you plan to collect over several periods, you have to remind that each scrapping is just adding up to the other ones. \
  Once the data scrapped, it is required to extract the informations:
  ```bash
  cd data
  python transform_scrapped_data.py -n my_beautiful_data
  ```
  It will be stored in the same folder.\
  Note: the scrapper usually works during roughly 2 days, then it needs to be restarted.


  ### For the trading environment training

  The same logic was implemented for the custom environment:
  ```bash
  cd custom_env
  python algo_comparison_JAX.py
  ```
  Similarly as before, the simulations will be stored in the folder algo_comparison/sim_number_... \
  Yet, this time, it is required to provide a data path for the training data, which aren't available on this GitHub repository for size issues, so one has to scrap them before. \
  This time, it is not recommended to run multiple threads at a time, since JAX uses the GPU more efficiently.

  Once you have found a parametrization that suits the results you want, you run an unique simulation with them:
  ```bash
  python run_DADAC.py
  ```
  This way, you'll be able to save the model (in the model folder for example).

  Notice: it is recommended to use distinct data for the training and the testing. One could train on the latest data one got and test on data gathered on another time period.

  ### For backtesting

  To run a backtest, one can use the following:
  ```bash
  python backtest_run.py
  ```
  It will propose 4 cases:
  + 0: run an offline backtest on local data. The data path has to be provided.
  + 1: run an offline backtest on online data. The algorithm will connect to the Binance websocket and run until it reaches a stop limit / loss or the number of required timesteps (1 timestep is roughly 100ms).
  + 2: run an online backtest on online data. This time, the algorithm the trades chosen by the algorithm are sent to Binance. The default trading value is around 6€ (corresponds to the minimum value of a trade). One has to provide its API keys in a .env file (BINANCE_API_KEY and BINANCE_API_SECRET). 
  + 3: bench the latency between the local computer and the Binance websocket.
  In every case, when using DADAC, one has to provide the path of a saved model. The default corresponds to an untrained model.


  ## Repo Structure

  ```text
  ├── backtest_run.py             # Script for running backtests (offline/online) and latency benchmarks
  ├── DSAC.py                     # Distributional SAC implementation
  ├── parallel_run.py             # Script for running parallel simulations for DADAC verification
  ├── SAC_VC.py                   # SAC with Value Correction implementation
  ├── custom_env/                 # Contains the JAX-based environment and RL algorithm implementations
  │   ├── algo_comparison_JAX.py  # Main script for training and comparing algorithms (SAC, DADAC, etc.)
  │   ├── discrete_policy.py      # Network definitions for discrete policies in JAX
  │   └── env.py                  # The trading environment logic, reward functions, and JAX optimizations
  │   └── run_DADAC.py            # Script to train a single DADAC model
  ├── data/                       # Data directory
  │   ├── transform_scrapped_data.py # Script to process raw scraped data
  │   └── scrapper/               # Tools to collect data from Binance
  │       ├── readme.txt          # Compilation and usage instructions for the scraper
  │       └── scrapper.cpp        # High-performance C++ WebSocket scraper
  ├── utils_backtest.py           # Helper functions for backtesting, plotting, and data management
  ├── utils_env.py                # Environment wrappers for latency simulation and buffering
  ├── utils_SAC.py                # Core SAC implementation details and custom wrappers
  └── README.md                   # This file
  ```

  ## Related Papers

* **DADAC (Delay Aware SAC)**: [*"Delay-Aware Soft Actor-Critic for Learning in the Presence of Delays"*](https://openreview.net/forum?id=Y9cVrdYn10) (Bheemaiah et al., 2020).  
  *Core reference for this project, introducing value correction and distributional approaches to mitigate delay-induced instability.*
* **DSAC (Distributional SAC)**: [*"Distributional Soft Actor-Critic"*](https://arxiv.org/abs/2004.14547) (Ma et al., 2020).  
  *Provides the framework for modeling Q-values as Gaussian distributions to improve sample efficiency and robustness.*

* **Stable-Baselines3**: [*"Reliable Reinforcement Learning Implementations"*](https://github.com/DLR-RM/stable-baselines3) (Raffin et al., 2021).  
  *The architectural standards of this repository follow the SB3 design patterns, adapted for JAX performance.*
