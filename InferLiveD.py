import pandas as pd
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from indicatorgen import calculate_indicators
from Xfeatures import extract_features
import os
import asyncio
import configparser
import threading

# ============================================================
# ---------------------- CONFIGURATION ------------------------
# ============================================================
model_data_pairs = {
    "CADCHF.keras": "CADCHF_otc_sorted.csv",
    "USDDZD.keras": "USDDZD_otc_sorted.csv",
    "USDPHP.keras": "USDPHP_otc_sorted.csv",
    "USDPKR.keras": "USDPKR_otc_sorted.csv"
}

LOG_FILE_SUFFIX = "_inference.csv"
LOOP_SLEEP_SECONDS = 5
INITIAL_LINES_TO_READ = 89
SUBSEQUENT_LINES_TO_READ = 59
WINDOW_SIZE = 40
INITIAL_INFERENCE_WINDOWS = 30

INI_FILE = "settingz.ini"


# ============================================================
# ------------------- ASYNC BACKGROUND TASK ------------------
# ============================================================
async def monitor_inference_csvs():
    """Runs every 5 seconds, checks last 30 lines of inference CSVs, updates INI."""
    print("Async monitor started.\n")

    while True:
        try:
            follow_set = set()
            invert_set = set()

            for model_filename, csv_filename in model_data_pairs.items():
                pair = csv_filename.split("_")[0]
                inference_file = f"{pair}_inference.csv"

                if not os.path.exists(inference_file):
                    continue

                # Read last 30 lines manually
                with open(inference_file, "r") as f:
                    lines = f.readlines()

                # Remove separators ******
                clean_lines = [l for l in lines if not l.startswith("*")]

                # Header + at least 1 row?
                if len(clean_lines) <= 1:
                    continue

                last_30 = clean_lines[-30:]
                good = 0
                bad = 0

                for line in last_30:
                    if "GOOD" in line:
                        good += 1
                    elif "BAD" in line:
                        bad += 1

                # Majority logic
                if good >= bad:
                    follow_set.add(pair)
                else:
                    invert_set.add(pair)

            # Now update settings.ini accordingly
            update_ini(follow_set, invert_set)

        except Exception as e:
            print(f"Async monitor error: {e}")

        await asyncio.sleep(5)


def update_ini(follow_set, invert_set):
    """Updates FOLLOW and INVERT assets in settingz.ini based on prediction majority."""
    config = configparser.ConfigParser()
    config.read(INI_FILE)

    # Convert sets to CSV lists
    follow_list = ",".join([p + "_otc" for p in follow_set])
    invert_list = ",".join([p + "_otc" for p in invert_set])

    config["FOLLOW"]["assets"] = follow_list
    config["INVERT"]["assets"] = invert_list

    with open(INI_FILE, "w") as f:
        config.write(f)

    print("\nUpdated settingz.ini:")
    print("FOLLOW:", follow_list if follow_list else "(none)")
    print("INVERT:", invert_list if invert_list else "(none)")


# ============================================================
# ---------------------- DATA PROCESSING ----------------------
# ============================================================
def get_samples_from_df(df):
    df_ind = calculate_indicators(df.copy())

    keep_cols = [
        'time', 'open', 'high', 'low', 'close',
        'bbwidth', 'adx', 'candle_size', 'upper_wick', 'lower_wick',
        'atr', 'macd_line', 'signal_line', 'macd_histogram'
    ]
    df_ind = df_ind[keep_cols]
    df_processed = df_ind.dropna().reset_index(drop=True)

    ydata = df_processed.drop(columns=['time'])
    time_data = df_processed['time']

    samples = []
    for i in range(len(ydata) - WINDOW_SIZE + 1):
        window = ydata.iloc[i:i+WINDOW_SIZE]
        window_time = time_data.iloc[i + WINDOW_SIZE - 1]

        class_x = window.to_numpy(dtype=np.float32)
        features = extract_features(window)
        features = [np.array(g, dtype=np.float32) for g in features]

        samples.append({
            'time': window_time,
            'class_x': class_x,
            'class_a': features[0],
            'class_b': features[1],
            'class_c': features[2],
            'class_d': features[3],
            'class_e': features[4],
            'class_f': features[5],
            'class_g': features[6],
        })

    return samples


# ============================================================
# ---------------------- INFERENCE ----------------------------
# ============================================================
def run_inference(model, samples):
    if not samples:
        return []

    inputs = {
        'main_input': np.stack([s['class_x'] for s in samples]),
        'side_input_1': np.stack([s['class_a'] for s in samples]),
        'side_input_2': np.stack([s['class_f'] for s in samples]),
        'side_input_3': np.stack([s['class_g'] for s in samples]),
        'side_input_4': np.stack([s['class_d'] for s in samples]),
        'side_input_5': np.stack([s['class_e'] for s in samples]),
        'side_input_6': np.stack([s['class_b'] for s in samples]),
        'side_input_7': np.stack([s['class_c'] for s in samples]),
    }

    preds = model.predict(inputs, verbose=0).flatten()
    return preds


# ============================================================
# ------------------- ORGANIZED LOGGING -----------------------
# ============================================================
def log_predictions(log_file, samples, predictions, is_first_write_this_session):
    file_exists = os.path.exists(log_file)
    add_separator = (is_first_write_this_session and file_exists)

    with open(log_file, 'a') as f:
        if add_separator:
            f.write("******\n")

        if not file_exists:
            f.write("time,prediction,classification\n")

        for sample, pred in zip(samples, predictions):
            cls = "GOOD" if pred >= 0.5 else "BAD"
            f.write(f"{sample['time']},{pred:.4f},{cls}\n")


# ============================================================
# -------------------------- MAIN -----------------------------
# ============================================================
def start_async_loop():
    """Runs asyncio loop in background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(monitor_inference_csvs())


if __name__ == "__main__":
    print("\n--- Loading Models ---")
    models = {}

    for model_path in model_data_pairs:
        if os.path.exists(model_path):
            try:
                models[model_path] = tf.keras.models.load_model(model_path)
                print(f"✅ Loaded: {model_path}")
            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")
        else:
            print(f"⚠️ Missing model: {model_path}")

    inference_state = {}
    for model_path, data_file in model_data_pairs.items():
        pair_name = data_file.split('_')[0]
        inference_state[pair_name] = {
            "last_processed_time": None,
            "is_first_run": True,
            "first_session_write": True
        }

    # -------- Start async background monitor --------
    thread = threading.Thread(target=start_async_loop, daemon=True)
    thread.start()

    print("\n--- Starting Live Inference Loop ---\n")

    # -------------------- MAIN LOOP --------------------------
    while True:
        for model_path, csv_file in model_data_pairs.items():
            pair_name = csv_file.split('_')[0]
            model = models.get(model_path)
            if model is None:
                continue

            print(f"\n--- Processing {pair_name} ---")

            state = inference_state[pair_name]
            lines_to_read = INITIAL_LINES_TO_READ if state["is_first_run"] else SUBSEQUENT_LINES_TO_READ

            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                abs_csv_path = os.path.join(script_dir, csv_file)

                df = pd.read_csv(abs_csv_path)
                if len(df) < lines_to_read:
                    print(f"Not enough data for {pair_name}: have {len(df)}, need {lines_to_read}")
                    continue
                df_tail = df.tail(lines_to_read)

            except FileNotFoundError:
                print(f"Missing CSV file: {abs_csv_path}")
                continue

            samples = get_samples_from_df(df_tail)
            if not samples:
                print("No samples produced.")
                continue

            samples_to_predict = []

            if state["is_first_run"]:
                samples_to_predict = samples[-INITIAL_INFERENCE_WINDOWS:]
                state["is_first_run"] = False
                print(f"Initial run: logging {len(samples_to_predict)} samples.")
            else:
                last_sample = samples[-1]
                if last_sample['time'] != state["last_processed_time"]:
                    samples_to_predict = [last_sample]
                    print(f"New candle found at {last_sample['time']}")
                else:
                    print("No new candle.")
                    continue

            predictions = run_inference(model, samples_to_predict)

            log_file = os.path.join(script_dir, f"{pair_name}{LOG_FILE_SUFFIX}")

            log_predictions(
                log_file,
                samples_to_predict,
                predictions,
                state["first_session_write"]
            )

            state["first_session_write"] = False
            state["last_processed_time"] = samples_to_predict[-1]['time']

            print(f"Logged {len(predictions)} to {log_file}")

        print(f"\n--- Loop complete. Sleeping {LOOP_SLEEP_SECONDS}s ---\n")
        time.sleep(LOOP_SLEEP_SECONDS)
