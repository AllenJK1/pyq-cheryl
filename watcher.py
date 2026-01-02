import os
import time
import subprocess

FILE_TO_WATCH = "USDPHP_otc.csv"
SCRIPT_TO_RUN = "finally3.py"
LOCK_FILE = "trade_active.lock"
CHECK_INTERVAL = 6  # How often we check the file


def is_script_running():
    """Check if the target script is currently running."""
    try:
        result = subprocess.run(f"pgrep -f {SCRIPT_TO_RUN}", shell=True, capture_output=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking if script is running: {e}")
        return False


def monitor_and_restart():
    """
    Monitors the file and restarts the script if idle too long.
    - If lock exists -> 3 minutes timeout.
    - If lock missing -> 15 sec timeout + guaranteed additional 9 sec (24 sec total minimum).
    """
    if not is_script_running():
        print(f"{SCRIPT_TO_RUNG} is not running. Please start it in another terminal.")

    extra_wait_no_lock = 9  # <-- NEW RULE

    while True:
        try:
            # Determine idle timeout
            if os.path.exists(LOCK_FILE):
                max_idle_time = 3 * 60
                print(f"Lock file found. Using long timeout: {max_idle_time}s.")
            else:
                # Add the new required 9 seconds
                max_idle_time = 15 + extra_wait_no_lock
                print(f"No lock file. Using short timeout + extra wait: {max_idle_time}s.")

            # Check file modification time
            if os.path.exists(FILE_TO_WATCH):
                last_modified_time = os.path.getmtime(FILE_TO_WATCH)
                time_difference = time.time() - last_modified_time

                print(f"Checking {FILE_TO_WATCH}: Last modified {int(time_difference)}s ago.")

                if time_difference > max_idle_time:
                    print(f"Idle > {max_idle_time}s. Terminating script...")

                    subprocess.run(f"pkill -f {SCRIPT_TO_RUN}", shell=True)
                    print("Script terminated. Waiting 65s for it to restart...")
                    time.sleep(65)
            else:
                print(f"Warning: {FILE_TO_WATCH} does not exist. Waiting...")

        except FileNotFoundError:
            print(f"Error: {FILE_TO_WATCH} not found. Waiting...")
        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    print(f"Starting watcher for {SCRIPT_TO_RUN}, checking file {FILE_TO_WATCH}.")
    monitor_and_restart()
