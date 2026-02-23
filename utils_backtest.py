import signal
import asyncio
import numpy as np
import websockets
import json
import os
os.chdir("./custom_env")
from custom_env.DADAC_JAX import DADAC_JAX
from custom_env.env import solve_constrained_oracle_pf, foolish_env
os.chdir("../")
import jax
import os
import gymnasium as gym
import time
import hmac
import hashlib
import aiohttp
import urllib.parse
import pandas as pd
from gymnasium.spaces import Box
from flax import serialization
import pickle
import matplotlib.pyplot as plt

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


plt.rcParams.update({
    "text.usetex": True
})

class ManualMarginTrader:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com"

    def _sign(self, query_string):
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method, endpoint, params):
        timestamp = time.time_ns()
        params['timestamp'] = int(timestamp / 1000)
        query_string = urllib.parse.urlencode(params)
        print(query_string, type(query_string))
        signature = self._sign(query_string)
        
        full_url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
        headers = {'X-MBX-APIKEY': self.api_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, full_url, headers=headers) as resp:
                # Check if the response is actually JSON
                content_type = resp.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    text = await resp.json()
                    return {"message": text, "timestamp": timestamp}
                else:
                    # Return the text or a status dict if it's not JSON
                    text = await resp.text()
                    return {"status": resp.status, "message": text, "timestamp": timestamp}
        
        return {"message": "", "timestamp": timestamp}

    async def margin_order(self, symbol, side, quantity, leverage_type="MARGIN_BUY"):
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'sideEffectType': leverage_type
        }
        return await self._request('POST', '/sapi/v1/margin/order', params)
    
    async def initialize_margin_account(self, asset="USDC", amount="1.0"):
        params = {
            'asset': asset,
            'amount': amount,
            'type': 1
        }
        return await self._request('POST', '/sapi/v1/asset/transfer', params)

class policy:
    """
    Policy types: benchmark | custom | test_send_unique | trained_policy -> pass 'params_path as kwargs | random | inactive
    """
    def __init__(self, kind, test_env, **kwargs):
        if kind == "benchmark":
            DISRESPECT_THE_ENV = foolish_env(20,1)
            self.model = DADAC_JAX("DiscretePolicy", DISRESPECT_THE_ENV, DISRESPECT_THE_ENV, None, None, None,
                              policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), target_entropy=1)
            with open(kwargs["param_path"], "rb") as f:
                serialized_bytes = f.read()
            template = {"actor": self.model.policy.actor_state.params}
            loaded_data = serialization.from_bytes(template, serialized_bytes)
            self.params = loaded_data["actor"]
            self.model.policy.actor_state = self.model.policy.actor_state.replace(params = self.params)
            self.waited_time = 0
            self.predict = self.predict_benchmark
        
        elif kind == "custom":
            self.predict = self.predict_custom

        elif kind == "test_send_unique":
            self.predict = self.predict_test
        
        elif kind == "trained_policy":
            self.predict = self.predict_RL
            DISRESPECT_THE_ENV = foolish_env(20,1)
            self.model = DADAC_JAX("DiscretePolicy", DISRESPECT_THE_ENV, DISRESPECT_THE_ENV, None, None, None,
                              policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn = jax.nn.gelu), target_entropy=1)
            with open(kwargs["param_path"], "rb") as f:
                serialized_bytes = f.read()
            template = {"actor": self.model.policy.actor_state.params}
            loaded_data = serialization.from_bytes(template, serialized_bytes)
            self.params = loaded_data["actor"]
            self.model.policy.actor_state = self.model.policy.actor_state.replace(params = self.params)
        
        elif kind == "random":
            self.predict = self.predict_random
        elif kind == "inactive":
            self.predict = self.predict_inactive
        else:
            raise "Not a correct policy"
        
        self.trade_counter = 0
        
    def predict_benchmark(self, obs):
        if self.waited_time < 10:
            self.waited_time += 1
            action = obs[15]
        else:
            self.waited_time = 0
            action = 1-obs[15]
            _ = self.predict_RL(obs)
        return action
        

    def predict_custom(self, obs):
        state = obs[15]
        mat_change = [[0,1,2], [0,1], [0,2]]
        prob_change = [[0.99,0.005,0.005], [0.01,0.99], [0.01,0.99]]
        action = np.random.choice(mat_change[int(state)], p = prob_change[int(state)])
        return action
    
    def predict_random(self, obs):
        state = obs[15]
        mat_change = [[0,1,2], [0,1], [0,2]]        
        action = np.random.choice(mat_change[int(state)])
        return action
    
    def predict_RL(self, obs):
        return self.model.policy._predict(obs.reshape(1,-1), deterministic=True)[0]
    
    def predict_test(self, obs):
        action = 0
        if obs[15] == 0 and self.trade_counter == 0:
            action = 1
            self.trade_counter += 1
        elif obs[15] == 1:
            action = 0
        return action
    
    def predict_inactive(self, obs):
        return 0


class Backtester():
    def __init__(self, test_type: str, policy_type: str, policy_kwargs: str, data_kwargs, max_window_length = 100, stop_loss = 0.98, stop_limit = 1.01, leverage = 1, max_ep_length = 1200, transaction_cost = 0.075/100,
                 private_key: str = None, public_key: str = None):
        self.test_type = test_type
        self.policy_kwargs = policy_kwargs
        self.max_window_length = max_window_length
        self.data_kwargs = data_kwargs
        
        self.rec_queue = asyncio.Queue()
        self.pred_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()
        self.state_queue = asyncio.Queue()
        self.send_data = asyncio.Queue()

        self.stop_loss = stop_loss
        self.stop_limit = stop_limit

        self.leverage = leverage
        self.max_ep_length = max_ep_length
        self.transaction_cost = transaction_cost

        ### Modifies the function according to the kind of test asked
        if test_type == "local":
            assert "data_path" in data_kwargs and "seed" in data_kwargs, "No data path provided"
            self.load_data_local(data_kwargs["data_path"], data_kwargs["seed"])
            self.preprocess_data = self.preprocess_data_local
            self.postprocess_data = self.postprocess_data_offline
            self.binance_scraper = self.binance_scraper_std
        elif test_type == "offline":
            self.preprocess_data = self.preprocess_data_online
            self.postprocess_data = self.postprocess_data_offline
            self.binance_scraper = self.binance_scraper_std
        elif test_type == "online":
            self.preprocess_data = self.preprocess_data_online
            self.postprocess_data = self.postprocess_data_online
            if policy_type != "benchmark":
                self.binance_scraper = self.binance_scraper_std
            else:
                self.binance_scraper = self.binance_scraper_backtest
            self.private_key = private_key
            self.public_key = public_key
            self.data_sender = ManualMarginTrader(public_key, private_key)
        else:
            raise NotImplementedError
    
        test_env = foolish_env(20,1)
        self.policy = policy(policy_type, test_env, **policy_kwargs)
        
        self.action_rec = np.zeros((max_ep_length + 1))
        self.obs_rec = np.zeros((max_ep_length + 1, 20))
        self.timestamp_rec = np.zeros((max_ep_length + 1, 5), dtype=np.uint64)

    def load_data_local(self, path, seed):
        np.random.seed(seed)
        data = pd.read_csv(path).to_numpy()
        arr = []
        idx = np.random.randint(0, data.shape[0]) + np.arange((self.max_ep_length+1 + self.max_window_length)*20)
        for pos in idx:
            if data[pos, 2] == 0:
                arr.append(data[pos, 3:23].reshape((2,5,2)))
            elif data[pos, 2] == 1: #o c h l
                dico = {"o": data[pos, 43], "c": data[pos, 44], "h": data[pos, 45], "l": data[pos, 46]}
                arr.append(dico)

        self.data = arr

    async def _init_state(self, ref_bid, ref_ask):
        await self.state_queue.put(np.array([(ref_bid + ref_ask)/2, (ref_bid + ref_ask)/2, 0.0, 1.0, 0.0, 1-self.stop_loss, self.stop_limit-1]))

    def _exit_condition(self, obs, action, potential_pf_value, timestep):
        return potential_pf_value < self.stop_loss or potential_pf_value > self.stop_limit or timestep > self.max_ep_length

    async def _step(self, obs, action, timestep):
        '''
        Returns a new observation according to the past observation and the action taken
        '''
        state = obs[15]
        pf_value = obs[16]
        current = obs[14]
        init = obs[13]
        
        idx = 3.0*action + state
        fees = self.transaction_cost * pf_value * self.leverage
        diff_norm = (current - init) / (init + 100) * pf_value * self.leverage
        inv_diff = (init - current) / (current + 100) * pf_value * self.leverage
        pf_diff = (idx == 1.0) * diff_norm + (idx == 2.0) * inv_diff 
        potential_pf_value = pf_value - fees * np.isin(idx, np.asarray([1.0, 2.0, 3.0, 6.0])) + pf_diff

        end_flag = self._exit_condition(obs, action, potential_pf_value, timestep)
        final_action = (1-end_flag) * action
        idx = 3.0*action + state
        new_pf_value = pf_value - fees * np.isin(idx, np.asarray([1.0, 2.0, 3.0, 6.0])) + pf_diff

        new_market_val = (idx == 0.0) * (obs[0] + obs[5]) / 2 + np.isin(idx, np.asarray([2.0,3.0,8.0])) * obs[5]+ np.isin(idx, np.asarray([1.0,4.0,6.0])) * obs[0]
        flag = action == state
        new_timestep_entry = flag * self.timestep_entry + (1-flag) * timestep
        new_entry_obs = flag * init + (1-flag) * new_market_val

        self.timestep_entry = new_timestep_entry

        await self.state_queue.put(np.array([new_entry_obs, new_market_val, final_action, new_pf_value, (timestep - new_timestep_entry)/self.max_ep_length, new_pf_value - self.stop_loss, self.stop_limit - new_pf_value]).reshape(-1))
        return final_action, end_flag

    async def local_scrapper(self):
        idx = 0
        await self.send_data.put(True)
        while True: 
            flag = await self.send_data.get()
            if flag:
                await self.rec_queue.put(self.data[idx])
                idx += 1
                flag = False

    async def preprocess_data_local(self):
        """
        Chooses data from an local file data episode, and shapes it as a policy observation
        """
        await asyncio.gather(*[self.local_scrapper(), self.data_normalizer()])

    async def preprocess_data_online(self):
        """
        Receives the data from binance servers, and shapes it as a policy observation
        """
        try:
            await self.data_sender.initialize_margin_account()
        except:
            pass
        await asyncio.gather(*[self.binance_scraper(), self.data_normalizer()])

    async def binance_scraper_std(self):
        streams = [
            "btcusdt@depth20@100ms", 
            "btcusdt@kline_1s"
        ]
        uri = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to Binance WebSocket...")
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    stream_name = data.get('stream')
                    payload = data.get('data')

                    if "depth" in stream_name:
                        bid = payload['bids'] # Best Bid Price
                        ask = payload['asks'] # Best Ask Price
                        
                        await self.rec_queue.put((bid, ask))

                    elif "kline" in stream_name:
                        k = payload['k']
                        await self.rec_queue.put((k))

        except Exception as e:
            print(f"Error: {e}")

    async def binance_scraper_backtest(self):
        streams = [
            "btcusdt@depth@100ms", 
            "btcusdt@kline_1s"
        ]
        uri = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        bid = np.array([[90000, 0.01], [90001, 0.01], [90002, 0.01], [90003, 0.01], [90004, 0.01]])
        ask = np.array([[89999, 0.01], [89998, 0.01], [89997, 0.01], [89996, 0.01], [89995, 0.01]])
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to Binance WebSocket...")
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    stream_name = data.get('stream')
                    payload = data.get('data')

                    if "depth" in stream_name:
                        timestamp = payload['E']
                        await self.rec_queue.put((bid, ask, timestamp))

                    elif "kline" in stream_name:
                        k = payload['k']
                        await self.rec_queue.put((k))

        except Exception as e:
            print(f"Error: {e}")

    async def data_normalizer(self):
        it = 0
        storing_vec = np.zeros((self.max_window_length,2), dtype= np.float32)
        vec_partial_obs = np.zeros(20, dtype=np.float32)
        self.ref_bid = 1.0
        self.ref_ask = 1.0
        self.ref_mean = 1.0
        flag_first = True
        flag_first_bis = True
        while True:
            data = await self.rec_queue.get()
            timestamp_reception = time.time_ns()
            if it < self.max_window_length:
                if len(data) == 4:
                    vec_partial_obs, storing_vec = self.compute_vec_partial_obs(0, vec_partial_obs, self.ref_bid, self.ref_ask, self.ref_mean, storing_vec, data, it % self.max_window_length)

                elif len(data) == 2:
                    data = np.array(data, dtype= np.float32)
                    if flag_first:
                        self.ref_bid = data[0,0,0]
                        self.ref_ask = data[1,0,0]
                        self.ref_mean = (self.ref_bid + self.ref_ask)/2
                        self.trade_quantity_base = 6 / self.ref_mean #* self.leverage : if one REALLY want to use leverage. please don't
                        flag_first = False

                    vec_partial_obs, storing_vec = self.compute_vec_partial_obs(1, vec_partial_obs, self.ref_bid, self.ref_ask, self.ref_mean, storing_vec, data, it % self.max_window_length)
                    it += 1
                if self.test_type == "local":
                    await self.send_data.put(True)

                elif len(data) == 3:
                    it += 1
                    data = np.array([data[0], data[1]], dtype= np.float32)
                    if flag_first:
                        self.ref_bid = data[0,0,0]
                        self.ref_ask = data[1,0,0]
                        self.ref_mean = (self.ref_bid + self.ref_ask)/2
                        self.trade_quantity_base = 6 / self.ref_mean #* self.leverage : if one REALLY want to use leverage. please don't
                        flag_first = False
                    vec_partial_obs, storing_vec = self.compute_vec_partial_obs(1, vec_partial_obs, self.ref_bid, self.ref_ask, self.ref_mean, storing_vec, data, it % self.max_window_length)

            else:
                if len(data) == 4:
                    vec_partial_obs, storing_vec = self.compute_vec_partial_obs(0, vec_partial_obs, self.ref_bid, self.ref_ask, self.ref_mean, storing_vec, data, it % self.max_window_length)
                    if self.test_type == "local":
                        await self.send_data.put(True)

                elif len(data) == 2:
                    data = np.array(data, dtype= np.float32)

                    vec_partial_obs, storing_vec = self.compute_vec_partial_obs(1, vec_partial_obs, self.ref_bid, self.ref_ask, self.ref_mean, storing_vec, data, it % self.max_window_length)
                    if flag_first_bis:
                        await self._init_state(vec_partial_obs[0], vec_partial_obs[5])
                        flag_first_bis = False
                        running_timestep = 0
                        self.timestep_entry = 0
                    it += 1
                    vec_partial_obs[-7:] = await self.state_queue.get()
                    running_timestep += 1
                    await self.pred_queue.put((vec_partial_obs, running_timestep, timestamp_reception, 0))

                elif len(data) == 3:
                    timestamp_server = data[2]
                    data = np.array([data[0], data[1]], dtype= np.float32)

                    vec_partial_obs, storing_vec = self.compute_vec_partial_obs(1, vec_partial_obs, self.ref_bid, self.ref_ask, self.ref_mean, storing_vec, data, it % self.max_window_length)
                    if flag_first_bis:
                        await self._init_state(vec_partial_obs[0], vec_partial_obs[5])
                        flag_first_bis = False
                        running_timestep = 0
                        self.timestep_entry = 0
                    it += 1
                    vec_partial_obs[-7:] = await self.state_queue.get()
                    running_timestep += 1
                    
                    await self.pred_queue.put((vec_partial_obs, running_timestep, timestamp_reception, timestamp_server))
                    

    def compute_vec_partial_obs(self, caso, past_vec_partial_obs, ref_bid, ref_ask, ref_mean, past_storing_vec, data, it):
        vec_partial_obs = np.zeros_like(past_vec_partial_obs)
        SHAPE = 100
        if caso == 0:
            open_val = (float(data["o"])/ref_mean - 1) * 100
            close_val = (float(data["c"])/ref_mean - 1) * 100
            spread = np.log10(float(data["h"]) - float(data["l"]) + 1)
            vec_partial_obs[10:13] = [open_val, close_val, spread]
        elif caso == 1:
            zero_mean_100_bid = past_storing_vec[it, 0]
            bid_val = data[0,0,0]/ref_bid - 1
            past_storing_vec[it, 0] = past_storing_vec[(it-1)%SHAPE, 0] + bid_val - 1
            bid_vol = np.sum(data[0,:5,1])
            mean10_bid = 100 * ((past_storing_vec[it, 0] - past_storing_vec[(it-10)%SHAPE, 0])/10 + 1)
            mean50_bid = 100 * ((past_storing_vec[it, 0] - past_storing_vec[(it-50)%SHAPE, 0])/50 + 1)
            mean100_bid = 100 * ((past_storing_vec[it, 0] - zero_mean_100_bid) / SHAPE + 1)

            zero_mean_100_ask = past_storing_vec[it, 1]
            ask_val = data[1,0,0]/ref_ask - 1
            past_storing_vec[it, 1] = past_storing_vec[(it-1)%SHAPE, 1] + ask_val - 1
            ask_vol = np.sum(data[1,:5,1])
            mean10_ask = 100 * ((past_storing_vec[it, 1] - past_storing_vec[(it-10)%SHAPE, 1])/10 + 1)
            mean50_ask = 100 * ((past_storing_vec[it, 1] - past_storing_vec[(it-50)%SHAPE, 1])/50 + 1)
            mean100_ask = 100 * ((past_storing_vec[it, 1] - zero_mean_100_ask) / SHAPE + 1)

            vec_partial_obs[:10] = [bid_val, bid_vol, mean10_bid, mean50_bid, mean100_bid, 
                                    ask_val, ask_vol, mean10_ask, mean50_ask, mean100_ask]

        return vec_partial_obs, past_storing_vec

    async def store_step(self, action, obs, timestep):
        self.action_rec[timestep] = action
        self.obs_rec[timestep] = obs
        self.max_timestep = timestep

    async def predict(self):
        '''
        Receives the preprocessed observation, and gets the result through the policy
        '''
        while True:
            obs, running_timestep, timestamp_reception, timestamp_server = await self.pred_queue.get()
            action = self.policy.predict(obs)
            final_action, done = await self._step(obs, action, running_timestep)
            current_state = obs[15]
            if not done:
                timestamp_end_action = time.time_ns()
                await self.output_queue.put((final_action, current_state, timestamp_reception, timestamp_end_action, timestamp_server, running_timestep, obs[16]))
            else:
                raise asyncio.CancelledError
            await self.store_step(final_action, obs, running_timestep)

    async def postprocess_data_offline(self):
        '''
        Takes a decision according to the action predicted by the policy, and sends it to data_dealer
        '''
        while True:
            action, current_state, timestamp_reception, timestamp_end_action, timestamp_server, running_timestep, pf_val = await self.output_queue.get()
            timestamp_sending = 0
            timestamp_order_application = 0
            await self.store_timestamps(timestamp_reception, timestamp_end_action, timestamp_sending, timestamp_order_application, running_timestep, timestamp_server)
            await self.send_data.put(True)

    def transform_action_to_message(self, action, current_state):
        idx = 3*action + current_state
        flag_send_order = np.isin(idx, [1,2,3,6])
        sell_side = None
        match idx:
            case 3: 
                sell_side = "BUY"
            case 6:
                sell_side = "BUY"
            case 1: 
                sell_side = "SELL"
            case 2:
                sell_side = "SELL"
        return flag_send_order, sell_side

    async def store_timestamps(self, timestamp_reception, timestamp_end_action, timestamp_sending, timestamp_order_application, timestep, timestamp_server):
        self.timestamp_rec[timestep] = [timestamp_server, timestamp_reception, timestamp_end_action, timestamp_sending, timestamp_order_application]


    async def postprocess_data_online(self):
        '''
        Takes a decision according to the action predicted by the policy, and sends it to data_dealer
        '''
        while True:
            action, current_state, timestamp_reception, timestamp_end_action, timestamp_server, running_timestep, pf_val = await self.output_queue.get()
            flag_send_order, sell_side = self.transform_action_to_message(action, current_state)
            if flag_send_order:
                qty = "{:.5f}".format(self.trade_quantity_base * pf_val)
                print(sell_side, qty)
                dic_return = await self.data_sender.margin_order("BTCUSDC", sell_side, qty)
                timestamp_sending = dic_return["timestamp"]
                try:
                    timestamp_order_application = dic_return["message"]["transactTime"]                    
                    await self.store_timestamps(timestamp_reception, timestamp_end_action, timestamp_sending, timestamp_order_application, running_timestep, timestamp_server)
                except:
                    print("error")
                
                

    async def run(self):
        '''
        Combines all the step from the reading of the shared memory to predicting an action to writing in the shared memory
        '''
        self.stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        
        def shutdown():
            for task in asyncio.all_tasks(loop):
                task.cancel()

        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, shutdown)
        
        tasks = [self.preprocess_data(), self.predict(), self.postprocess_data()]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("End of simulation")

        return self.action_rec, self.obs_rec, self.timestamp_rec, self.max_timestep, self.ref_mean

def plot_benchmark(load_path, save_path, fees):
    with open(load_path, 'rb') as f:
        x = pickle.load(f)

    scaled_data = (x["observations"][0][1:int(x["fly_time"]) + 1,0]/100 + 1)* x["ref"]

    opt_act, opt_pf = solve_constrained_oracle_pf(x["observations"][0][1:int(x["fly_time"]) + 1,0], fees, 1.0)

    opt_act = opt_act
    opt_pf = opt_pf

    re_act =x["observations"][0][1:int(x["fly_time"]) + 1,15]
    re_pf =x["observations"][0][1:int(x["fly_time"]) + 1,16]

    fig, axs = plt.subplots(1,3, figsize = (18,5))

    x_grid = np.linspace(0, len(opt_act) / 10, len(opt_act))

    axs[0].plot(x_grid, re_act, label = "Algorithm actions")
    axs[0].plot(x_grid, opt_act, label = "Afterwards optimal actions")
    axs[0].grid()
    axs[0].set_title("Actions taken")    
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Action (0: neutral, 1: long, 2: short)")
    axs[0].legend()

    axs[1].plot(x_grid, re_pf, label = "Algorithm portfolio")
    axs[1].plot(x_grid, opt_pf, label = "Afterwards optimal portfolio")
    axs[1].grid()
    axs[1].set_title("Simulated portfolio values")    
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Portfolio values (1: no variation)")
    axs[1].legend()
    

    axs[2].plot(x_grid, (scaled_data / scaled_data[1] - 1)*100, label = "EUR / BTC")
    axs[2].plot([0, len(scaled_data)/10], [fees, fees], '--', c = 'r', label = "Trading fees")
    axs[2].plot([0, len(scaled_data)/10], [-fees, -fees], '--', c = 'r')
    axs[2].grid()
    axs[2].set_title("Market variations")    
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Market variations (in \%, relative to $t=0$)")
    axs[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi = 300)