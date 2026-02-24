import numpy as np
import pandas as pd
import sys


def extract_order_book(df, mean_window=10):
    lines = df[df["data_type"]==0.0]
    
    cumulated_quantity_bids = np.sum(lines[["v1", "v3", "v5", "v7", "v9",]].to_numpy(), axis = 1)
    cumulated_quantity_asks = np.sum(lines[["v11", "v13", "v15", "v17", "v19",]].to_numpy(), axis = 1)

    bids = lines["v0"].to_numpy()
    asks = lines["v10"].to_numpy()

    cumsum_bids = np.cumsum(bids)
    cumsum2_bids = np.cumsum(bids**2)
    cumsum_asks = np.cumsum(asks)
    cumsum2_asks = np.cumsum(asks ** 2)

    running_mean_bids = (cumsum_bids[mean_window:] - cumsum_bids[:-mean_window]) / mean_window
    running_mean_asks = (cumsum_asks[mean_window:] - cumsum_asks[:-mean_window]) / mean_window
    
    running_std_bids = ((cumsum2_bids[mean_window:] - cumsum2_bids[:-mean_window])/(mean_window - 1) - running_mean_bids ** 2) ** 0.5
    running_std_asks = ((cumsum2_asks[mean_window:] - cumsum2_asks[:-mean_window])/(mean_window - 1) - running_mean_asks ** 2) ** 0.5

    return bids, cumulated_quantity_bids, running_mean_bids, running_std_bids, asks, cumulated_quantity_asks, running_mean_asks, running_std_asks

def extract_candles_vals(df):
    lines = df[df["data_type"]==1.0][["v40", "v41", "v42", "v43", "v44", "v45", "v46"]].to_numpy()

    open = lines[:,0]
    high = lines[:,1]
    low = lines[:,2]
    close = lines[:,3]
    volume = lines[:, 4]
    share = lines[:, 5]
    
    return open, high, low, close, volume, share

def global_np_writer(df):

    mean_window_1 = 10
    mean_window_2 = 50
    mean_window_3 = 100

    bids, cumulated_quantity_bids, running_mean10_bids, running_std10_bids, asks, cumulated_quantity_asks, running_mean10_asks, running_std10_asks = extract_order_book(df, mean_window=mean_window_1)
    _, _, running_mean50_bids, running_std50_bids, _, _, running_mean50_asks, running_std50_asks = extract_order_book(df, mean_window=mean_window_2)
    _, _, running_mean100_bids, running_std100_bids, _, _, running_mean100_asks, running_std100_asks = extract_order_book(df, mean_window=mean_window_3)
    open, high, low, close, volume, share = extract_candles_vals(df)

    pos_first_timestep = df[df["data_type"] == 1.0]["counter"].iloc[0]
    first_timestep = df[df["data_type"] == 1.0]["timestamp"].iloc[0]
    
    bids = bids[mean_window_3 + pos_first_timestep - 1:]
    asks = asks[mean_window_3 + pos_first_timestep - 1:]
    cumulated_quantity_asks = cumulated_quantity_asks[mean_window_3 + pos_first_timestep - 1:]
    cumulated_quantity_bids = cumulated_quantity_bids[mean_window_3 + pos_first_timestep - 1:]
    
    running_mean10_asks = running_mean10_asks[mean_window_3 - mean_window_1 + pos_first_timestep - 1:]
    running_mean10_bids = running_mean10_bids[mean_window_3 - mean_window_1 + pos_first_timestep - 1:]
    running_mean50_asks = running_mean50_asks[mean_window_3 - mean_window_2 + pos_first_timestep - 1:]
    running_mean50_bids = running_mean50_bids[mean_window_3 - mean_window_2 + pos_first_timestep - 1:]
    running_mean100_asks = running_mean100_asks[pos_first_timestep - 1:]
    running_mean100_bids = running_mean100_bids[pos_first_timestep - 1:]
    
    running_std10_asks = running_std10_asks[mean_window_3 - mean_window_1 + pos_first_timestep - 1:]
    running_std10_bids = running_std10_bids[mean_window_3 - mean_window_1 + pos_first_timestep - 1:]
    running_std50_asks = running_std50_asks[mean_window_3 - mean_window_2 + pos_first_timestep - 1:]
    running_std50_bids = running_std50_bids[mean_window_3 - mean_window_2 + pos_first_timestep - 1:]
    running_std100_asks = running_std100_asks[pos_first_timestep - 1:]
    running_std100_bids = running_std100_bids[pos_first_timestep - 1:]

    init_pos_candles = np.where(df['data_type'] == 1.0)[0]
    pos_candles = init_pos_candles-init_pos_candles[0]
    pos_candles -= np.arange(len(pos_candles))
    pos_candles = np.concatenate([pos_candles, [len(bids)]])
    
    high_adapted = np.empty_like(bids)
    low_adapted = np.empty_like(bids)
    close_adapted = np.empty_like(bids)
    open_adapted = np.empty_like(bids)

    for k in range(len(pos_candles)-1):
        high_adapted[pos_candles[k] : pos_candles[k+1]] = high[k]
        low_adapted[pos_candles[k] : pos_candles[k+1]] = low[k]
        close_adapted[pos_candles[k] : pos_candles[k+1]] = close[k]
        open_adapted[pos_candles[k] : pos_candles[k+1]] = open[k]

    ret = np.vstack([bids, cumulated_quantity_bids, running_mean10_bids, running_mean50_bids, running_mean100_bids, running_std10_bids, running_std50_bids, running_std100_bids, 
                          asks, cumulated_quantity_asks, running_mean10_asks, running_mean50_asks, running_mean100_asks, running_std10_asks, running_std50_asks, running_std100_asks,
                          open_adapted, close_adapted, high_adapted, low_adapted]).T
    return ret


if __name__ == '__main__':
    args = sys.argv[1:]
    
    if "-n" in args[0]:
        out_name = args[1]
    else:
        out_name = "transformed_data"

    path = "scrapper/scrapper_out.csv"
    df = pd.read_csv(path)
    ret = global_np_writer(df)
    np.save( out_name + ".npy", ret)
