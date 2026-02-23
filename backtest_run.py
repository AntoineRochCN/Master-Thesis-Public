from utils_backtest import *
import asyncio
import time 
import pickle
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

def backtest(policy_type: str, policy_kwargs, max_ep_length, N_tests, leverage, experience_name, data_path = "./data/test_data.csv"):
    test_type = "local"
    
    act_rec_arr = []
    obs_rec_arr = []
    timestamp_rec_arr = []
    time_of_flight_rec_arr = []
    durations = np.zeros(N_tests)

    for k in range(N_tests):
        seed = k
        data_kwargs = {"seed": seed, "data_path": data_path}    
        obj = Backtester(test_type, policy_type, policy_kwargs, data_kwargs, max_ep_length=max_ep_length, public_key=api_key, private_key = api_secret, leverage=leverage)

        t1 = time.time()
        act_rec, obs_rec, timestamp_rec, time_of_flight, ref = asyncio.run(obj.run())
        durations[k] = time.time() - t1
        act_rec_arr.append(act_rec)
        obs_rec_arr.append(obs_rec)
        timestamp_rec_arr.append(timestamp_rec)
        time_of_flight_rec_arr.append(time_of_flight)

    act_rec_arr = np.array(act_rec_arr)
    obs_rec_arr = np.array(obs_rec_arr)
    timestamp_rec_arr = np.array(timestamp_rec_arr)
    time_of_flight_rec_arr = np.array(time_of_flight_rec_arr)

    file_name = "./backtest/backtest/" + experience_name + ".pkl"
    output_dict = {"actions" : act_rec_arr, "observations": obs_rec_arr, "timestamps": timestamp_rec_arr, "fly_time": time_of_flight_rec_arr, "durations": durations, "ref": ref}

    with open(file_name, "wb") as f:
        pickle.dump(output_dict, f)

    plot_benchmark(file_name, "./backtest/backtest/" + experience_name  + '.png', fees=0.075/100)

def offline_test(policy_type: str, policy_kwargs, max_ep_length, N_tests, leverage, experience_name):
    test_type = "offline"
    data_kwargs = {}
    
    act_rec_arr = []
    obs_rec_arr = []
    timestamp_rec_arr = []
    time_of_flight_rec_arr = []
    durations = np.zeros(N_tests)

    for k in range(N_tests):
        obj = Backtester(test_type, policy_type, policy_kwargs, data_kwargs, max_ep_length=max_ep_length, public_key=api_key, private_key = api_secret, leverage=leverage)

        t1 = time.time()
        act_rec, obs_rec, timestamp_rec, time_of_flight, ref = asyncio.run(obj.run())
        durations[k] = time.time() - t1
        act_rec_arr.append(act_rec)
        obs_rec_arr.append(obs_rec)
        timestamp_rec_arr.append(timestamp_rec)
        time_of_flight_rec_arr.append(time_of_flight)

    act_rec_arr = np.array(act_rec_arr)
    obs_rec_arr = np.array(obs_rec_arr)
    timestamp_rec_arr = np.array(timestamp_rec_arr)
    time_of_flight_rec_arr = np.array(time_of_flight_rec_arr)

    file_name = "./backtest/offline_test/" + experience_name  + ".pkl"
    output_dict = {"actions" : act_rec_arr, "observations": obs_rec_arr, "timestamps": timestamp_rec_arr, "fly_time": time_of_flight_rec_arr, "durations": durations, "ref": ref}

    with open(file_name, "wb") as f:
        pickle.dump(output_dict, f)

    plot_benchmark(file_name, "./backtest/offline_test/" + experience_name  + '.png', fees=0.075/100)

def online_test(policy_type: str, policy_kwargs, max_ep_length, N_tests, leverage, experience_name):
    test_type = "online"
    data_kwargs = {}

    act_rec_arr = []
    obs_rec_arr = []
    timestamp_rec_arr = []
    time_of_flight_rec_arr = []
    durations = np.zeros(N_tests)

    for k in range(N_tests):
        obj = Backtester(test_type, policy_type, policy_kwargs, data_kwargs, max_ep_length=max_ep_length, public_key=api_key, private_key = api_secret, leverage=leverage)

        t1 = time.time()
        act_rec, obs_rec, timestamp_rec, time_of_flight, ref = asyncio.run(obj.run())
        durations[k] = time.time() - t1
        act_rec_arr.append(act_rec)
        obs_rec_arr.append(obs_rec)
        timestamp_rec_arr.append(timestamp_rec)
        time_of_flight_rec_arr.append(time_of_flight)

    act_rec_arr = np.array(act_rec_arr)
    obs_rec_arr = np.array(obs_rec_arr)
    timestamp_rec_arr = np.array(timestamp_rec_arr)
    time_of_flight_rec_arr = np.array(time_of_flight_rec_arr)

    file_name = "./backtest/online_test/" + experience_name  + ".pkl"
    output_dict = {"actions" : act_rec_arr, "observations": obs_rec_arr, "timestamps": timestamp_rec_arr, "fly_time": time_of_flight_rec_arr, "durations": durations, "ref": ref}

    with open(file_name, "wb") as f:
        pickle.dump(output_dict, f)

    plot_benchmark(file_name, "./backtest/online_test/" + experience_name  + '.png', fees=0.075/100)

def bench_latency(max_trade_num ,policy_kwargs, experience_name):
    test_type = "online"
    policy_type = "benchmark"
    data_kwargs = {}
    max_ep_length = max_trade_num
    obj = Backtester(test_type, policy_type, policy_kwargs, data_kwargs, max_ep_length=max_ep_length, public_key=api_key, private_key = api_secret, stop_loss=0.3, leverage=leverage)
    
    act_rec, obs_rec, timestamp_rec, time_of_flight, ref = asyncio.run(obj.run())

    file_name = "./backtest/benchmark/" + experience_name  + ".pkl"
    output_dict = {"actions" : act_rec, "observations": obs_rec, "timestamps": timestamp_rec, "fly_time": time_of_flight, "ref": ref}

    with open(file_name, "wb") as f:
        pickle.dump(output_dict, f)


if __name__ == '__main__':
    test_case = eval(input("What to do? \n0: backtest\n1: offline testing\n2: online testing\n3: latency benchmark\n"))
    
    try:
        policy_type = eval(input("\nWhich policy: (0, DADAC) (1, random) (2, unique - test) (3, inactive): ")) 
        policy_arr = ["trained_policy", "random", "test_send_unique", 'inactive']
        policy_type = policy_arr[policy_type]
    except:
        pass
    
    ep_length = eval(input("Episode length (recommended: 1200): "))
    N_tests = eval(input("Number of tests: "))
    leverage = eval(input("Leverage: "))
    experience_name = input("Save file: ")

    print("In case of using a DADAC model, be sure to provide a valid saved model path")
    saved_model_path = "./saved_models/model_test.msgpack"
    policy_kwargs = {"param_path": saved_model_path}

    print("In case of using a local data, be sure to provide a valid data path")
    data_path = "./data/test_data.csv" 
    match test_case:
        case 0:
            backtest(policy_type, policy_kwargs, ep_length, N_tests, leverage, experience_name, data_path = data_path)
        case 1:
            offline_test(policy_type, policy_kwargs, ep_length, N_tests, leverage, experience_name)
        case 2:
            online_test(policy_type, policy_kwargs, ep_length, N_tests, leverage, experience_name)
        case 3:
            bench_latency(ep_length, policy_kwargs, experience_name)