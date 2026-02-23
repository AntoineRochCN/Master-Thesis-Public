# Managing latencies in RL using a SAC variant

## Project Presentation
This repository is the support of my master's thesis over latencies management in reinforcement learning (RL) using the DADAC variant of the SAC algorithm, published in the [DADAC article](https://openreview.net/forum?id=Y9cVrdYn10).

The subject falls into two distinct parts:
- First, since the DADAC paper was rejected from ICLR 2025 and the code wasn't available anymore, I had to implement it again to verify their hipotesis and results.
- Once the method reviewed, the other objective was to apply it to a new field. Latency management in RL is often related to the robotic area, but I didn't have any robot to hand, so I tackled the automatic trading latencies problem.

## Verifying DADAC's results

To ensure their results were correct, I based my work on the Stable-Baselines3 library, and its fork SBX. 
The key points to navigate in the repository are the following:
- Four algorithms are to be compared, SAC, SAC_VC (SAC + Value Correction), DSAC (Distributional SAC) and DADAC (Delay Aware SAC). They all rely on eponym files, with a common structure set in the utils_SAC.py file. \
Their main differences stand in the critic loss computation:
  - DSAC changes the critic network to compute the parameters of a gaussian distribution, instead of a scalar value. It then adds a loss based on the variance of the distribution.
  - Value Correction is a variant of the n-step bootstrapping method (averages the $n$ forward observations critic values), where the weights correspond to the observed latency distribution (observation + action)  
  - DADAC is the mix of both methods
- To simulate the latency, wrappers were introduced in the utils_env.py file. To do so, all the latencies a pre-computed when reseting the environment, and the actions are chosen / applied only when the timestep corresponds to $t + \tau_{\text{observation}}$ (time to compute an action) and $t + \tau_{\text{observation}} + t_{\text{action}}$ (time to apply the action to the environment). \
There are many buffers in the file and the implementation is prone to small bugs (one timestep shift etc...), yet it's not so impactful over the results. \
Between two timesteps corresponding to the application of an action over the environment, the algorithm assumes that the past action is still beeing applied to the environment.
- To run many simulations with varying parametrizations, one can use parallel_run.py.


## Implementing the trading environment

All the functions related to the internal operations of the trading environment and the algorithms related belong to the custom_env folder.

### JAX details

To implement the trading environment, I fully switched to JAX. Thus, the environment, the wrappers and the RL algorithms entirely run on a GPU, which increase the overall speed of the algorithm by roughly x40, without parallelizing the environments. However, it implies more memory consumption since all the buffers were set on the GPU, and the training metrics are only available at the end of a training.\
This led to some adaptations to the algorithm:
- One loop of collecting observations makes one critic update. The policy is updated every $n_{\text{policy}}$ times the critic is updated. The metrics are recorded every $n_{\text{record}}$
times the policy is updated, and the policy is testes every $n_{\text{test}}$ times the environment is recorded. Consequently, one has to be careful when choosing the parametrization. \
This is for performance improvements, since is simplifies by a lot the computation of the XLA graph, and produce a +37% speed increase, without losing the SAC logic. 
- JAX struggles to updates arrays in-place when on the GPU. It requires fixed-size arrays and doesn't behave very well with the stochastic behaviors. Consequently, some parts of the code do more iterations than required (stop a while loop for example), in order to stay with fixed size items.
- The policy always compute the action from the observation, even if the timestep doesn't correspond to the latency. This is to avoid a JAX if statement, which slowers more the computation by beeing a burden on the XLA graph than going through a MLP (256,256,256). However, when dealing with bigger data, it might bring a slowdown.

### Environment details

The environment is based on Binance websockets values, especially the order book sent every 100ms.
The data used for training was scrapped from it, and consequently has its format. The other data used was the 1s candlesticks from the same API.

The normalization is detailed in the env.py file, as for the environment step behavior. The reward function can be tuned in the reward_function.py file. The buffers also changed a little to be JAX compatible, but kept the same structure (buffer.py).
The data is almost trained raw:
  - Only the first values of bid and ask are kept. They are normalized within the first value of the episode. The volume corresponds to the sum of the first 5 bid / ask values.
  - The portfolio value begins at 1, so an ending value of 0.95 means a 5% loss.
  - The trading fees and leverage can be adapted.
  - The "oracle" function corresponds to the maximum value a portfolio can reach by the end of an episode, computed within dynamic programming. It is very useful to measure how good the algorithm behaves (the volatility may sometimes not be sufficient to generate profit).
  - Staying neutral refers to the action 0, going long to 1 and going short to 2.
  - Since the action space is discrete ($\{0, 1, 2\}$), the SAC had to be modified to predict discrete actions. This leads to some variations in the output of the critic and policy networks, along with other kind of behavior of the losses / the entropy.
  - Understanding how trading works is a hard task for a RL algorithm, and it struggles generating good observations. To deal with that problem, a first warmup step is set overfitting the environment over the best actions found within dynamic programming.

## Repo Structure

```text
├── data/               # Raw and processed datasets (ignored if large)
├── docs/               # Documentation and GitHub Pages assets
├── models/             # Trained model checkpoints or logic
├── notebooks/          # Jupyter notebooks for exploratory analysis
├── scripts/            # Utility Python scripts
├── .env.example        # Template for Binance API keys
├── .gitignore          # Prevents .env and data leaks
├── main.py             # Entry point of the application
└── requirements.txt    # Project dependencies
```

## Installation

jax 0.6.2 doesn't behave very well with nvidia-cublas-cu12 12.8, but torch requires this version. Hence, we need to overwrite the package with the newer version.

```bash
pip install -r requirements.txt
pip install --no-deps --force-reinstall nvidia-cublas-cu12>=12.9.1.4
```


## 📝 Related Papers

1.  **Author, A., & Author, B. (202X).** *Title of the most relevant paper*. Journal Name. [Link]
2.  **Binance Research.** *Market Microstructure Report*. [Link]
3.  **Your Name (2026).** *Preliminary Findings on Crypto Volatility*. Thesis Working Paper.