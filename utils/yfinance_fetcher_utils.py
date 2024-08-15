import os
import lzma
import pytz
import pandas as pd
import dill as pickle

from datetime import datetime
from typing import Tuple, List, Dict


def get_tickers_and_ticker_dfs(
    data_fetcher,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Fetches tickers and their corresponding historical data from Yahoo Finance.

    Returns:
        Tuple[List[str], Dict[str, pd.DataFrame]]: A tuple containing a list of tickers and a dictionary where the keys are tickers and the values are DataFrames with historical data.
    """
    period_start = datetime(2023, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)

    obj_path = "yfinance_dataset.obj"

    tickers, ticker_dfs = data_fetcher.get_ticker_dfs(
        start=period_start, end=period_end, obj_path=obj_path
    )

    return tickers, ticker_dfs


def load_pickle(path: str):
    """
    Loads and returns a Python object from a pickled file.

    Args:
        path (str): The file path to the pickled file.

    Returns:
        Any: The Python object stored in the pickled file.
    """
    with lzma.open(path, "rb") as fp:
        file = pickle.load(fp)
    return file


def save_pickle(path: str, obj):
    """
    Saves a Python object to a file using pickle.

    Args:
        path (str): The file path where the object should be saved.
        obj (Any): The Python object to be pickled and saved.

    Returns:
        None
    """
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    with lzma.open(path, "wb") as fp:
        pickle.dump(obj, fp)
