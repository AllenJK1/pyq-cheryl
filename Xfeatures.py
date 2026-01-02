import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression

def adx_and_market_strength(data_window: pd.DataFrame):
    adx_series = data_window['adx']
    adx_mean = adx_series.mean()
    adx_above_30_ratio = (adx_series > 30).sum() / len(adx_series)
    adx_stc = adx_series.iloc[-1] - adx_series.iloc[0]
    adx_std = adx_series.std()
    return adx_mean, adx_above_30_ratio, adx_stc, adx_std

def entropy_and_chaos(candles: pd.DataFrame):
    price_diff = candles['close'].diff().dropna()
    price_direction = (price_diff > 0).astype(int)
    price_entropy = entropy(price_direction.value_counts(normalize=True), base=2)

    macd_diff = candles['macd_histogram'].diff().dropna()
    macd_direction = (macd_diff > 0).astype(int)
    macd_entropy = entropy(macd_direction.value_counts(normalize=True), base=2)

    body_size = candles['candle_size'].dropna() + 1e-8
    quantized = pd.qcut(body_size, q=4, duplicates='drop', labels=False)
    candle_body_entropy = entropy(quantized.value_counts(normalize=True), base=2)

    return price_entropy, macd_entropy, candle_body_entropy

def gap_range_and_price_levels(data_window: pd.DataFrame):
    highs = data_window['high']
    lows = data_window['low']
    opens = data_window['open']
    closes = data_window['close']

    high_low_range = highs.max() - lows.min()
    total_range = (highs - lows).sum()
    total_body = (closes - opens).abs().sum()
    range_to_body_ratio = total_range / total_body if total_body != 0 else np.nan

    previous_closes = closes.shift(1)
    gap_candles = (opens != previous_closes).sum()
    gap_candle_ratio = gap_candles / (len(data_window) - 1)

    return high_low_range, range_to_body_ratio, gap_candle_ratio

def macd_based_momentum_summary(data_window: pd.DataFrame):
    macd_hist = data_window['macd_histogram']
    macd_hist_mean = macd_hist.mean()
    macd_hist_std = macd_hist.std()
    macd_crosses = ((macd_hist.shift(1) * macd_hist) < 0).sum()
    macd_bullish_ratio = (macd_hist > 0).sum() / len(macd_hist)
    macd_bearish_ratio = (macd_hist < 0).sum() / len(macd_hist)
    return macd_hist_mean, macd_hist_std, macd_crosses, macd_bullish_ratio, macd_bearish_ratio

def price_action_and_trend_features(data_window: pd.DataFrame):
    close_prices = data_window['close'].values
    open_prices = data_window['open'].values
    candle_sizes = data_window['candle_size'].values

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    trend_slope = model.coef_[0][0]
    trend_strength = model.score(X, y)

    trend_changes = np.sign(np.diff(close_prices))
    direction_changes = np.count_nonzero(np.diff(trend_changes) != 0)

    bullish = (close_prices > open_prices)
    bearish = (close_prices < open_prices)
    doji = (np.isclose(close_prices, open_prices, atol=1e-5))

    bullish_ratio = bullish.sum() / len(close_prices)
    bearish_ratio = bearish.sum() / len(close_prices)
    doji_ratio = doji.sum() / len(close_prices)

    mean_body = np.mean(candle_sizes)
    strong_threshold = mean_body * 1.5
    strong_bull = ((bullish) & (candle_sizes > strong_threshold)).sum()
    strong_bear = ((bearish) & (candle_sizes > strong_threshold)).sum()

    strong_bull_ratio = strong_bull / len(close_prices)
    strong_bear_ratio = strong_bear / len(close_prices)

    return (
        trend_slope, trend_strength, direction_changes,
        bullish_ratio, bearish_ratio, strong_bull_ratio,
        strong_bear_ratio, doji_ratio
    )

def volatility_and_noise(data_window: pd.DataFrame):
    avg_candle_size = data_window['candle_size'].mean()
    candle_size_std = data_window['candle_size'].std()
    atr_mean = data_window['atr'].mean()
    atr_std = data_window['atr'].std()
    bbwidth_mean = data_window['bbwidth'].mean()
    bbwidth_std = data_window['bbwidth'].std()
    return avg_candle_size, candle_size_std, atr_mean, atr_std, bbwidth_mean, bbwidth_std

def wick_behavior_and_volatility(data_window: pd.DataFrame):
    upper_wick = data_window['upper_wick']
    lower_wick = data_window['lower_wick']
    body_size = data_window['candle_size']
    body_size_safe = body_size.replace(0, 1e-8)

    upper_wick_ratio = upper_wick / body_size_safe
    lower_wick_ratio = lower_wick / body_size_safe

    avg_upper_wick_to_body_ratio = upper_wick_ratio.mean()
    avg_lower_wick_to_body_ratio = lower_wick_ratio.mean()
    upper_wick_dominance_ratio = (upper_wick > lower_wick).sum() / len(data_window)
    lower_wick_dominance_ratio = (lower_wick > upper_wick).sum() / len(data_window)
    wick_symmetry = (upper_wick_ratio - lower_wick_ratio).abs()
    wick_symmetry_std = wick_symmetry.std()

    return (
        avg_upper_wick_to_body_ratio,
        avg_lower_wick_to_body_ratio,
        upper_wick_dominance_ratio,
        lower_wick_dominance_ratio,
        wick_symmetry_std
    )

def extract_features(window: pd.DataFrame):
    class_a = np.array(adx_and_market_strength(window), dtype=np.float32)     # ADX
    class_b = np.array(entropy_and_chaos(window), dtype=np.float32)           # Entropy
    class_c = np.array(gap_range_and_price_levels(window), dtype=np.float32)  # Gaps and range
    class_d = np.array(macd_based_momentum_summary(window), dtype=np.float32) # MACD momentum
    class_e = np.array(price_action_and_trend_features(window), dtype=np.float32) # Trend + price action
    class_f = np.array(volatility_and_noise(window), dtype=np.float32)        # Noise/Volatility
    class_g = np.array(wick_behavior_and_volatility(window), dtype=np.float32) # Wick behavior

    return class_a, class_b, class_c, class_d, class_e, class_f, class_g
