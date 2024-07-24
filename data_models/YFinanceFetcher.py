import os
import pytz
import logging
import yfinance
import pandas as pd

from datetime import datetime
from data_models.DataFetcher import DataFetcher
from utils.alpha_utils import load_pickle, save_pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


class YFinanceFetcher(DataFetcher):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_history(self, ticker, period_start, period_end, granularity="1d", tries=0):
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
            self.logger.info(
                f"YFinanceFetcher - ERROR - get_history() - {ex} - num of tries - {tries}"
            )
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

    def get_histories(self, tickers, start, end, granularity="1d"):
        period_starts = [start] * len(tickers)
        period_ends = [end] * len(tickers)
        dfs = [None] * len(tickers)

        def _helper(i):
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

    def get_ticker_dfs(self, start, end, obj_path):
        dataset_path = os.path.join(os.getcwd(), "Data", obj_path)
        dataset_exists = os.path.exists(dataset_path)

        crypto_asset_config = "yfinance_crypto_asset_config.json"
        tickers = super().get_crypto_tickers(crypto_asset_config)

        # If .obj file doesn't exists, download all data
        if not dataset_exists:
            new_tickers, new_dfs = self.get_histories(
                tickers, start, end, granularity="1d"
            )
            new_ticker_dfs = {ticker: df for ticker, df in zip(new_tickers, new_dfs)}
            save_pickle(dataset_path, (tickers, new_ticker_dfs))
            return new_tickers, new_ticker_dfs
        # If .obj file exists, load existing data
        else:
            old_tickers, old_ticker_dfs = load_pickle(dataset_path)

            # If we added new tickers to config file, then download history data for that tickers
            new_tickers = list(set(tickers) - set(old_tickers))
            if new_tickers:
                new_tickers, new_dfs = self.get_histories(
                    new_tickers, start, end, granularity="1d"
                )
                new_ticker_dfs = {
                    ticker: df for ticker, df in zip(new_tickers, new_dfs)
                }

                for ticker in new_tickers:
                    old_ticker_dfs[ticker] = new_ticker_dfs[ticker]

                save_pickle(dataset_path, (list(old_ticker_dfs.keys()), old_ticker_dfs))
            last_date_index = old_ticker_dfs["BTC-USD"].index[-1]
            last_date = last_date_index.strftime("%Y-%m-%d")
            today_date = datetime.now(pytz.utc).strftime("%Y-%m-%d")

            # If we have one day more time script then return the same data (do not fetch data more times for the same day)
            if last_date == today_date:
                return list(old_ticker_dfs.keys()), old_ticker_dfs
            else:
                start = last_date_index.to_pydatetime()
                new_tickers, new_dfs = self.get_histories(
                    tickers, start, end, granularity="1d"
                )
                new_ticker_dfs = {
                    ticker: df for ticker, df in zip(new_tickers, new_dfs)
                }

                for asset_pair in new_ticker_dfs:
                    old_ticker_dfs[asset_pair] = old_ticker_dfs[asset_pair].drop(
                        old_ticker_dfs[asset_pair].tail(1).index
                    )
                    old_ticker_dfs[asset_pair] = pd.concat(
                        [old_ticker_dfs[asset_pair], new_ticker_dfs[asset_pair]]
                    )
                save_pickle(dataset_path, (list(old_ticker_dfs.keys()), old_ticker_dfs))
                return list(old_ticker_dfs.keys()), old_ticker_dfs
