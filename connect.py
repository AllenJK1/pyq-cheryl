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

# Load credentials
email, password = credentials()

# Initialize client
client = Quotex(email=email, password=password, lang="pt")

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
            print("Already connected!")
            return True
    return False

async def main():
    connected = await connect()
    if not connected:
        print("Failed to connect after retries.")
        sys.exit(1)

    # Keep the connection alive (optional)
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected by user.")
