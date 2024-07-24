import os
import json


class DataFetcher:

    def get_crypto_tickers(self, json_file):
        # load from json list
        crypto_asset_config_path = os.path.join(os.getcwd(), "config", json_file)
        with open(crypto_asset_config_path, "r") as f:
            crypto_tickers = json.load(f)
        return crypto_tickers["instruments"]
