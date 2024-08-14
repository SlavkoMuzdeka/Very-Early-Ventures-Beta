import os
import json

from typing import Optional, List


class DataFetcher:

    def get_crypto_tickers(self, json_path_file: str) -> Optional[List[str]]:
        """
        Loads and returns a list of cryptocurrency tickers from a JSON file.

        Args:
            json_path_file (str): The name of the JSON file in the "config" directory.

        Returns:
            list: Cryptocurrency tickers from the "instruments" key, or None if not found.
        """
        crypto_asset_config_path = os.path.join(os.getcwd(), "config", json_path_file)
        with open(crypto_asset_config_path, "r") as f:
            crypto_tickers = json.load(f)
            if crypto_tickers["instruments"]:
                return crypto_tickers["instruments"]
        return None
