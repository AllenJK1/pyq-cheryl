import os
import sys
import json
import time
import asyncio
import random
import configparser
import pandas as pd
import numpy as np
from datetime import datetime
from quotexapi.stable_api import Quotex
from quotexapi.config import credentials
email, password = credentials()


# --- Default Strategy Parameters ---
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Adjusted oversold level for CALL
RSI_OVERBOUGHT = 70  # Adjusted overbought level for PUT
KELTNER_PERIOD = 20
KELTNER_MULTIPLIER = 2
CONFIRMATION_CANDLES = 5  # Number of candles for confirmation
BBANDS_PERIOD = 20
BBANDS_STDDEV = 2
STOCHASTIC_K = 14
STOCHASTIC_D = 3
STOCHASTIC_J = 3
ADX_PERIOD = 14
ADX_THRESHOLD = 30  # Strong trend threshold for ADX

# --- File Paths ---
SETTINGS_FILE = "settingz.ini"

# --- Create or Load Settings ---
def initialize_settings():
    config = configparser.ConfigParser()
    if not os.path.exists(SETTINGS_FILE):
        config['ACCOUNT'] = {
            'ACCOUNT_TYPE': 'PRACTICE'
        }
        config['TRADE'] = {
            'TRADE_AMOUNT': '1',
            'TRADE_DURATION': '60'
        }
        config['ASSETS'] = {
            'ASSET_LIST': 'USDARS_otc'
        }
        config['FOLLOW'] = {
            'assets': 'USDPKR_otc,USDDZD_otc'
        }
        config['INVERT'] = {
            'assets': 'CADCHF_otc,USDPHP_otc'
        }
        with open(SETTINGS_FILE, 'w') as configfile:
            config.write(configfile)
    config.read(SETTINGS_FILE)
    return config

settings = initialize_settings()

# --- Read Settings ---
ACCOUNT_TYPE = settings['ACCOUNT']['ACCOUNT_TYPE']
TRADE_AMOUNT = float(settings['TRADE']['TRADE_AMOUNT'])
TRADE_DURATION = int(settings['TRADE']['TRADE_DURATION'])
ASSETS = settings['ASSETS']['ASSET_LIST'].split(',')
CANDLE_PERIOD = 5  # Candle period (5 seconds)

# --- Client Initialization ---
client = Quotex(email=email, password=password, lang="pt")
trade_active = False  # Tracks if a trade is ongoing

# --- Utility Functions ---
async def connect():
    """Ensure connection to Quotex API."""
    for _ in range(5):
        if not await client.check_connect():
            connected, reason = await client.connect()
            if connected:
                print("Connected successfully!")
                return True
            print(f"Connection failed: {reason}")
            await asyncio.sleep(5)
        else:
            return True
    return False

import csv
import os
import time

last_fetch_time = time.time()

async def fetch_candles(asset, count, timeframe=CANDLE_PERIOD):
    """Fetch historical candles and append them to a CSV file."""
    global last_fetch_time
    try:
        current_time = time.time()
        candles = await client.get_candles(asset, current_time, count, timeframe)
        if candles is None:
            raise Exception("Failed to fetch candles")

        # Prepare the file path
        file_path = f"{asset}.csv"

        # Check if the file exists, and create it if it doesn't
        file_exists = os.path.isfile(file_path)

        # Open the file in append mode
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write headers if file doesn't exist
            if not file_exists:
                writer.writerow(["time", "open", "high", "low", "close"])

            # Append the fetched candles to the file
            for candle in candles:
                writer.writerow([candle['time'], candle['open'], candle['high'], candle['low'], candle['close']])

        last_fetch_time = time.time()  # Update timestamp on success
        return candles
    except Exception as e:
        print(f"Error fetching candles for {asset}: {e}")
        raise


def validate_conditions(rsi, keltner, bbands, stochastic, adx, last_candle):
    """Validate all strategy conditions using the last candle in the analysis set."""
    keltner_upper, keltner_lower, close_price = keltner
    bbands_upper, bbands_lower, _ = bbands
    stochastic_k, stochastic_d, stochastic_j = stochastic

    # Check conditions for CALL
    if (close_price < keltner_lower and rsi < RSI_OVERSOLD and
        close_price < bbands_lower and stochastic_k < 20 and stochastic_d < 20 and stochastic_j < 20 and
        adx > ADX_THRESHOLD):
        return "call"

    # Check conditions for PUT
    elif (close_price > keltner_upper and rsi > RSI_OVERBOUGHT and
          close_price > bbands_upper and stochastic_k > 80 and stochastic_d > 80 and stochastic_j > 80 and
          adx > ADX_THRESHOLD):
        return "put"

    return None

def confirm_trade(candles, direction, indicators):
    """Confirm trade with the last 5 confirmation candles."""
    confirmation_candles = candles[-CONFIRMATION_CANDLES:]  # Last 5 candles for confirmation
    rsi, stochastic, cci, bbands, adx = indicators

    # Condition 1: At least 3 of the 5 candles close bullish or bearish
    bullish_count = sum(1 for c in confirmation_candles if c['close'] > c['open'])
    bearish_count = sum(1 for c in confirmation_candles if c['close'] < c['open'])

    if direction == "call":
        if bullish_count < 3:
            return False

    elif direction == "put":
        if bearish_count < 3:
            return False

    # Condition 2: RSI, Stochastic, and CCI maintain conditions
    if direction == "call":
        if not (rsi < RSI_OVERSOLD and stochastic[0] < 20 and stochastic[1] < 20 and cci < -100):
            return False

    elif direction == "put":
        if not (rsi > RSI_OVERBOUGHT and stochastic[0] > 80 and stochastic[1] > 80 and cci > 100):
            return False

    # Condition 3: ADX remains strong
    if adx <= ADX_THRESHOLD:
        return False

    # Condition 4: Price does not touch counter-trend zones for all confirmation candles
    bbands_upper, bbands_lower, _ = bbands
    for candle in confirmation_candles:
        if direction == "call" and candle['close'] >= bbands_upper:
            return False
        if direction == "put" and candle['close'] <= bbands_lower:
            return False

    return True
def calculate_rsi(candles):
    """Calculate RSI based on candle data."""
    gains, losses = [], []
    for i in range(1, len(candles)):
        change = candles[i]['close'] - candles[i-1]['close']
        gains.append(max(0, change))
        losses.append(abs(min(0, change)))
    avg_gain = sum(gains[-RSI_PERIOD:]) / RSI_PERIOD
    avg_loss = sum(losses[-RSI_PERIOD:]) / RSI_PERIOD
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_keltner_channels(candles):
    """Calculate Keltner Channels."""
    closes = pd.Series([c['close'] for c in candles])
    ema = closes.rolling(KELTNER_PERIOD).mean()
    atr = pd.Series([abs(c['high'] - c['low']) for c in candles]).rolling(KELTNER_PERIOD).mean()
    upper_channel = ema + (KELTNER_MULTIPLIER * atr)
    lower_channel = ema - (KELTNER_MULTIPLIER * atr)
    return upper_channel.iloc[-1], lower_channel.iloc[-1], closes.iloc[-1]

def calculate_bbands(candles):
    """Calculate Bollinger Bands."""
    closes = pd.Series([c['close'] for c in candles])
    sma = closes.rolling(BBANDS_PERIOD).mean()
    std_dev = closes.rolling(BBANDS_PERIOD).std()
    upper_band = sma + (BBANDS_STDDEV * std_dev)
    lower_band = sma - (BBANDS_STDDEV * std_dev)
    return upper_band.iloc[-1], lower_band.iloc[-1], closes.iloc[-1]
def calculate_adx_values(candles, period=14):
    """
    Calculate ADX values for a given set of candles.
    Expects at least 29 candles to ensure the last 15 candles have valid ADX.
    """
    if len(candles) < period + 15:  # Ensure enough candles to calculate ADX
        return None

    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    closes = np.array([c['close'] for c in candles])

    plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1] - lows[1:], 0)

    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(period) / period, mode='valid')  # Average True Range

    plus_di = 100 * (np.convolve(plus_dm, np.ones(period) / period, mode='valid') / atr)
    minus_di = 100 * (np.convolve(minus_dm, np.ones(period) / period, mode='valid') / atr)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.convolve(dx, np.ones(period) / period, mode='valid')

    return adx[-15:]  # Return the last 15 ADX values


def calculate_adx_mean(adx_values):
    """
    Calculate the average ADX over the last 15 candles.
    """
    if adx_values is None or len(adx_values) < 15:
        return None
    return sum(adx_values) / 15  # Average of the last 15 ADX values
def calculate_stochastic(candles):
    """Calculate Stochastic Oscillator."""
    closes = pd.Series([c['close'] for c in candles])
    highs = pd.Series([c['high'] for c in candles])
    lows = pd.Series([c['low'] for c in candles])
    k = ((closes - lows.rolling(STOCHASTIC_K).min()) / (highs.rolling(STOCHASTIC_K).max() - lows.rolling(STOCHASTIC_K).min())) * 100
    d = k.rolling(STOCHASTIC_D).mean()
    j = (3 * k) - (2 * d)
    return k.iloc[-1], d.iloc[-1], j.iloc[-1]

def calculate_cci(candles):
    """Calculate CCI."""
    closes = pd.Series([c['close'] for c in candles])
    highs = pd.Series([c['high'] for c in candles])
    lows = pd.Series([c['low'] for c in candles])
    tp = (highs + lows + closes) / 3
    sma_tp = tp.rolling(20).mean()
    mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci.iloc[-1]

def calculate_adx(candles):
    """Calculate ADX."""
    highs = pd.Series([c['high'] for c in candles])
    lows = pd.Series([c['low'] for c in candles])
    closes = pd.Series([c['close'] for c in candles])
    plus_dm = highs.diff()
    minus_dm = lows.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([highs - lows, (highs - closes.shift(1)).abs(), (lows - closes.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ADX_PERIOD).mean()
    plus_di = (plus_dm.rolling(ADX_PERIOD).sum() / atr) * 100
    minus_di = (minus_dm.abs().rolling(ADX_PERIOD).sum() / atr) * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.rolling(ADX_PERIOD).mean()
    return adx.iloc[-1]
async def get_candles(asset):
    """Reads the last 40 lines from the sorted CSV file for the asset."""
    candles = []
    try:
        # Open the CSV file for the given asset
        with open(f'{asset}_sorted.csv', mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)

            # Skip the header
            rows = rows[1:]

            # Get the last 40 rows (candles)
            candles = rows[-40:]

            # Convert each row into a dictionary or a list of appropriate values (time, open, high, low, close)
            # The format will be [time, open, high, low, close]
            candles = [
                {
                    'time': row[0],
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                }
                for row in candles
            ]

    except Exception as e:
        print(f"Error reading {asset}_sorted.csv: {e}")
    return candles

def load_asset_settings():
    """Load asset settings from settingz.ini."""
    config = configparser.ConfigParser()
    config.read(SETTINGS_FILE)
    follow_assets = config.get('FOLLOW', 'assets').split(',')
    invert_assets = config.get('INVERT', 'assets').split(',')
    return follow_assets, invert_assets

import subprocess
import time

# Termux vibration function
import asyncio

# Async Termux vibration function
import asyncio

async def trigger_vibration(mode):
    """Triggers vibration on Termux if available."""
    try:
        # Check if 'termux-vibrate' command is available
        if subprocess.run(["which", "termux-vibrate"], capture_output=True).returncode == 0:
            duration_ms = 0
            if mode == 1:  # Single Short
                duration_ms = 200
            elif mode == 2:  # Double Short
                await asyncio.create_subprocess_exec("termux-vibrate", "-d", "200")
                await asyncio.sleep(0.5)
                await asyncio.create_subprocess_exec("termux-vibrate", "-d", "200")
                return
            elif mode == 3:  # Short + Long
                await asyncio.create_subprocess_exec("termux-vibrate", "-d", "500")
                await asyncio.sleep(0.5)
                await asyncio.create_subprocess_exec("termux-vibrate", "-d", "1000")
                return

            if duration_ms > 0:
                await asyncio.create_subprocess_exec("termux-vibrate", "-d", str(duration_ms))
        else:
            # Silently ignore if not on Termux or if termux-api is not installed
            pass
    except Exception as e:
        print(f"Vibration command failed: {e}")


# Modified async trade function
import datetime
import csv
import os
LOCK_FILE = "trade_active.lock"
LOG_FILE = "trades_log.csv"



# Ensure the log file has headers if it doesn't exist
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Time", "Asset", "Original Direction", "Final Direction",
            "Action Taken", "Trade Status", "Result Amount"
        ])
# =====================================================
#  LOCKFILE MANAGEMENT (single function with subfunctions)
lockfile_delete_task = None  # GLOBAL variable, define at top of file


async def lockfile_manager(action, lockfile):
    global lockfile_delete_task

    async def _create_lock():
        global lockfile_delete_task

        # Cancel existing delete task if still running
        if lockfile_delete_task is not None:
            if not lockfile_delete_task.done():
                lockfile_delete_task.cancel()
                print("Existing lockfile removal task cancelled.")
            lockfile_delete_task = None

        # Create the lock file
        with open(lockfile, "w") as f:
            pass
        print(f"Lock file {lockfile} created.")

    async def _remove_lock():
        await asyncio.sleep(10)

        if os.path.exists(lockfile):
            os.remove(lockfile)
            print(f"Lock file {lockfile} removed.")

    if action == "create":
        await _create_lock()

    elif action == "remove":
        # Schedule background removal task
        lockfile_delete_task = asyncio.create_task(_remove_lock())

    else:
        print("Unknown lockfile action:", action)

# =====================================================
#  TRADE FUNCTION (calls lockfile manager)
# =====================================================
async def place_trade(asset, direction):
    """Execute a trade and manage lock creation/removal externally."""

    # --------------------------
    # CREATE LOCK
    # --------------------------
    await lockfile_manager("create", LOCK_FILE)

    try:

        # Load asset settings
        follow_assets, invert_assets = load_asset_settings()

        # Time of trade
        trade_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        original_direction = direction
        action_taken = "No action taken."

        # Apply asset logic
        if asset in invert_assets:
            direction = "put" if direction == "call" else "call"
            action_taken = f"Inverted direction for asset {asset}."
        elif asset in follow_assets:
            action_taken = f"Followed original direction for asset {asset}."
        else:
            action_taken = f"No rule for asset {asset}; using original direction."

        print("\n--- TRADE INITIATED ---")
        print(f"Time: {trade_time}")
        print(f"Asset: {asset}")
        print(f"Original Direction: {original_direction.upper()}")
        print(f"Final Direction: {direction.upper()}")
        print(f"{action_taken}")
        print(f"Placing trade | Amount: {TRADE_AMOUNT} | Duration: {TRADE_DURATION}s")

        await trigger_vibration(1)

        # Execute the trade
        status, info = await client.buy(TRADE_AMOUNT, asset, direction, TRADE_DURATION)

        result_amount = "N/A"
        trade_status = "FAILED"

        if status and "id" in info:
            trade_id = info["id"]
            print("Trade placed. Waiting for result...\n")

            await asyncio.sleep(TRADE_DURATION + 10)

            result_amount = await client.check_win(trade_id)
            trade_status = "WON" if result_amount > 0 else "LOST"

            print("\n--- TRADE RESULT ---")
            print(f"Result: {trade_status} | Amount: {result_amount}")

            await trigger_vibration(2 if result_amount > 0 else 3)

        else:
            print("Trade failed to place.")
            sys.exit()

        # Log to CSV
        with open(LOG_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_time, asset, original_direction.upper(), direction.upper(),
                action_taken, trade_status, result_amount
            ])

    finally:
        # --------------------------
        # REMOVE LOCK
        # --------------------------
        await lockfile_manager("remove", LOCK_FILE)


async def analyze_and_trade():
    """Main function to analyze and trade."""
    global trade_active
    settings = initialize_settings()  # Re-read settings each loop
    ASSETS = settings['ASSETS']['ASSET_LIST'].split(',')
    TRADE_AMOUNT = float(settings['TRADE']['TRADE_AMOUNT'])

    for asset in ASSETS:
        if trade_active:
            print("Trade active. Waiting for it to finish...")
            await asyncio.sleep(1)
            continue

        print(f"Analyzing asset: {asset}")

        try:
            candles = await fetch_candles(asset, 30)
            candles1 = await get_candles(asset)
            analysis_candles = candles[:-CONFIRMATION_CANDLES]  # First 25 candles for analysis
            confirmation_candles = candles[-CONFIRMATION_CANDLES:]
              # Last 5 candles for confirmation

            # Calculate indicators
            rsi = calculate_rsi(analysis_candles)
            keltner = calculate_keltner_channels(analysis_candles)
            bbands = calculate_bbands(analysis_candles)
            stochastic = calculate_stochastic(analysis_candles)
            cci = calculate_cci(analysis_candles)
            adx = calculate_adx(analysis_candles)
            adx2 = calculate_adx(candles)

            # Validate conditions
            last_analysis_candle = analysis_candles[-1]
            direction = validate_conditions(rsi, keltner, bbands, stochastic, adx, last_analysis_candle)

            if direction:
                confirmed = confirm_trade(candles, direction, (rsi, stochastic, cci, bbands, adx))
                if confirmed:
                    await place_trade(asset, direction)
                else:
                    print(f"{direction.upper()} not confirmed. Skipping trade.")
            else:
                print("No trade conditions met.")

        except Exception as e:
            print(f"Error analyzing {asset}: {e}")

async def main_logic():
    """The main operational logic of the script."""
    if not await connect():
        raise SystemExit("Failed to connect to API. Restarting...")

    while True:
        if not trade_active:
            await analyze_and_trade()
        await asyncio.sleep(3)

def failure_notification():
    """Plays a sound or vibrates to signal a critical failure."""
    try:
        # Check for Termux environment by command availability
        if subprocess.run(["which", "termux-vibrate"], capture_output=True).returncode == 0:
            print("Termux environment detected. Triggering a long vibration for failure.")
            subprocess.run(["termux-vibrate", "-d", "5000"])
        else:
            # Assume desktop environment
            print("Desktop environment detected. Attempting to play failure sound.")
            try:
                import pygame
                pygame.init()
                if os.path.exists("failure.mp3"):
                    pygame.mixer.music.load("failure.mp3")
                    pygame.mixer.music.play()
                    print("Playing 'failure.mp3'.")
                    time.sleep(10)  # Give it time to play
                else:
                    print("Warning: 'failure.mp3' not found. Cannot play failure sound.")
            except ImportError:
                print("Warning: Pygame library not installed. 'pip install pygame'")
            except Exception as e:
                print(f"Error playing sound with Pygame: {e}")
    except Exception as e:
        print(f"An error occurred in the notification system: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main_logic())
    except (SystemExit, Exception) as e:
        print(f"A critical error occurred: {e}")
        failure_notification()
        print("Script has been shut down.")
