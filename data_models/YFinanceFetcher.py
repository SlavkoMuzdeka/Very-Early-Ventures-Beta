import os
import json
import pytz
import logging
import yfinance
import pandas as pd

from datetime import datetime
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


class YFinanceFetcher:

    def __init__(self):
        """
        Initializes the YFinanceFetcher with a logger.
        """
        self.logger = logging.getLogger(__name__)
        self.tickers = self.get_crypto_tickers()

    def get_crypto_tickers(self) -> Optional[List[str]]:
        """
        Loads and returns a list of cryptocurrency tickers from a JSON file.

        Returns:
            list: Cryptocurrency tickers from the "instruments" key, or None if not found.
        """
        crypto_asset_config_path = os.path.join(
            os.getcwd(), "config", "yfinance_crypto_asset_config.json"
        )
        with open(crypto_asset_config_path, "r") as f:
            crypto_tickers = json.load(f)
            if crypto_tickers["instruments"]:
                return crypto_tickers["instruments"]
        return None

    def get_history(
        self,
        ticker: str,
        period_start: datetime,
        period_end: datetime,
        granularity="1d",
        tries=0,
    ) -> pd.DataFrame:
        """
        Fetches historical data for a given ticker from Yahoo Finance.

        Args:
            ticker (str): The ticker symbol of the financial instrument.
            period_start (datetime): The start date for historical data.
            period_end (datetime): The end date for historical data.
            granularity (str): The granularity of the data (e.g., '1d', '1h').
            tries (int): The number of retry attempts in case of failure.

        Returns:
            pd.DataFrame: A DataFrame containing historical data with columns 'datetime', 'open', 'high', 'low', 'close', 'volume'.
        """
        try:
            df = (
                yfinance.Ticker(ticker)
                .history(
                    start=period_start,
                    end=period_end,
                    interval=granularity,
                    auto_adjust=True,
                )
                .reset_index()
            )
        except Exception as ex:
            self.logger.info(f"YFinanceFetcher - ERROR - get_history() - {ex}")
            if tries < 5:
                return self.get_history(
                    ticker, period_start, period_end, granularity, tries + 1
                )
            return pd.DataFrame()

        df = df.rename(
            columns={
                "Date": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if df.empty:
            return pd.DataFrame()

        df.datetime = pd.DatetimeIndex(df.datetime.dt.date)
        df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
        df = df.drop(columns=["Dividends", "Stock Splits"])
        df = df.set_index("datetime", drop=True)
        return df

    def get_histories(
        self, tickers: List[str], start: datetime, end: datetime, granularity="1d"
    ) -> Tuple[List[str], List[pd.DataFrame]]:
        """
        Fetches historical data for multiple tickers concurrently.

        Args:
            tickers (List[str]): A list of ticker symbols.
            start (datetime): The start date for historical data.
            end (datetime): The end date for historical data.
            granularity (str): The granularity of the data (e.g., '1d', '1h').

        Returns:
            Tuple[List[str], List[pd.DataFrame]]: A tuple containing a list of tickers and a list of DataFrames with historical data.
        """
        period_starts = [start] * len(tickers)
        period_ends = [end] * len(tickers)
        dfs = [None] * len(tickers)

        def _helper(i: int):
            return self.get_history(
                tickers[i], period_starts[i], period_ends[i], granularity=granularity
            )

        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_helper, i): i for i in range(len(tickers))}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    dfs[i] = future.result()
                except Exception as exc:
                    self.logger.info(
                        f"Ticker {tickers[i]} generated an exception: {exc}"
                    )

        tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
        dfs = [df for df in dfs if not df.empty]
        return tickers, dfs

    def get_ticker_dfs(
        self, start: datetime, end: datetime
    ) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """
        Retrieves historical data for tickers.

        Args:
            start (datetime): The start date for historical data.
            end (datetime): The end date for historical data.

        Returns:
            Tuple[List[str], Dict[str, pd.DataFrame]]: A tuple containing a list of tickers and a dictionary of DataFrames with historical data.
        """
        new_tickers, new_dfs = self.get_histories(
            self.tickers, start, end, granularity="1d"
        )
        new_ticker_dfs = {ticker: df for ticker, df in zip(new_tickers, new_dfs)}

        return new_tickers, new_ticker_dfs
