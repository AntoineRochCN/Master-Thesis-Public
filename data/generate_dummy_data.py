"""
Synthetic dummy data generator for Master-Thesis-Public.
Generates realistic geometric Brownian motion price series with bid/ask spread
and order book features formatted as required by EnvDataBinance.
"""

import numpy as np
from pathlib import Path


def generate_synthetic_market_data(n_timesteps: int = 20000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)

    # Base price process (Geometric Brownian Motion around BTC price ~80,000)
    dt = 0.1  # 100ms
    sigma = 0.002
    mu = 0.00001
    log_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=n_timesteps)
    mid_price = 80000.0 * np.exp(np.cumsum(log_returns))

    spread = np.random.uniform(0.5, 2.5, size=n_timesteps)
    bids = mid_price - spread / 2.0
    asks = mid_price + spread / 2.0

    cum_qty_bids = np.random.uniform(1.0, 10.0, size=n_timesteps)
    cum_qty_asks = np.random.uniform(1.0, 10.0, size=n_timesteps)

    # Running means and stds for bids
    def rolling_mean_std(x, window):
        cs = np.cumsum(x)
        cs2 = np.cumsum(x ** 2)
        mean = np.empty_like(x)
        std = np.empty_like(x)
        mean[:window] = cs[:window] / np.arange(1, window + 1)
        mean[window:] = (cs[window:] - cs[:-window]) / window
        std[:window] = np.sqrt(np.maximum(cs2[:window] / np.arange(1, window + 1) - mean[:window] ** 2, 1e-8))
        std[window:] = np.sqrt(np.maximum((cs2[window:] - cs2[:-window]) / window - mean[window:] ** 2, 1e-8))
        return mean, std

    rm10_bids, rstd10_bids = rolling_mean_std(bids, 10)
    rm50_bids, rstd50_bids = rolling_mean_std(bids, 50)
    rm100_bids, rstd100_bids = rolling_mean_std(bids, 100)

    rm10_asks, rstd10_asks = rolling_mean_std(asks, 10)
    rm50_asks, rstd50_asks = rolling_mean_std(asks, 50)
    rm100_asks, rstd100_asks = rolling_mean_std(asks, 100)

    # 1s candle values
    open_p = mid_price + np.random.normal(0, 0.2, size=n_timesteps)
    close_p = mid_price + np.random.normal(0, 0.2, size=n_timesteps)
    high_p = np.maximum(open_p, close_p) + np.abs(np.random.normal(0, 0.5, size=n_timesteps))
    low_p = np.minimum(open_p, close_p) - np.abs(np.random.normal(0, 0.5, size=n_timesteps))

    cols = [
        bids, cum_qty_bids, rm10_bids, rm50_bids, rm100_bids, rstd10_bids, rstd50_bids, rstd100_bids,
        asks, cum_qty_asks, rm10_asks, rm50_asks, rm100_asks, rstd10_asks, rstd50_asks, rstd100_asks,
        open_p, close_p, high_p, low_p
    ]
    data = np.column_stack(cols).astype(np.float32)
    return data


def main():
    data_dir = Path(__file__).resolve().parent

    train_data = generate_synthetic_market_data(n_timesteps=20000, seed=42)
    test_data = generate_synthetic_market_data(n_timesteps=20000, seed=123)

    train_path = data_dir / "transformed_train_data.npy"
    test_path = data_dir / "transformed_test_data.npy"

    np.save(train_path, train_data)
    np.save(test_path, test_data)

    print(f"Generated synthetic market data:")
    print(f"  Train: {train_path} shape: {train_data.shape}")
    print(f"  Test:  {test_path} shape: {test_data.shape}")


if __name__ == "__main__":
    main()
