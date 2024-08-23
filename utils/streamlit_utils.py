import os
import lzma
import pytz
import pandas as pd
import dill as pickle
import streamlit as st
import plotly.graph_objects as go

from alpha import Alpha
from datetime import datetime
from typing import List, Tuple, Dict
from data_models.YFinanceFetcher import YFinanceFetcher


def show_timeseries_plot(tickers: List[str], dfs_path: str):
    """
    Displays line plots for Alpha and Beta values for multiple tickers on the same plot.

    Args:
        tickers (List[str]): A list of ticker symbols for which to plot Alpha and Beta.
        dfs_path (str): The path to the file containing the DataFrames with Alpha and Beta data.
    """
    if not tickers:
        st.warning("Please select at least one asset to show timeseries plot!")
        return

    dfs = load_pickle(dfs_path)

    figures = [go.Figure(), go.Figure()]
    columns = ["beta_eth", "alpha_eth"]
    axis_names = ["Beta", "Alpha"]

    for i, fig in enumerate(figures):
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=dfs[ticker].index,
                    y=dfs[ticker][columns[i]],
                    mode="lines",
                    name=f"{ticker} {axis_names[i]}",
                )
            )

    for i, fig in enumerate(figures):
        fig.update_layout(
            title=f"{axis_names[i]} Values Over Time",
            xaxis_title="Date",
            yaxis_title=f"{axis_names[i]}",
            legend_title="Ticker",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)


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


def get_tickers_and_ticker_dfs() -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Fetches tickers and their corresponding historical data from Yahoo Finance.

    Returns:
        Tuple[List[str], Dict[str, pd.DataFrame]]: A tuple containing a list of tickers and a dictionary where the keys are tickers and the values are DataFrames with historical data.
    """
    period_start = datetime(2023, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)

    data_fetcher = YFinanceFetcher()

    tickers, ticker_dfs = data_fetcher.get_ticker_dfs(
        start=period_start, end=period_end
    )

    return tickers, ticker_dfs


def calculate_alpha(
    tickers: List[str],
    ticker_dfs: Dict[str, pd.DataFrame],
    window=60,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculates alpha and beta values for given tickers and stores the results.

    Args:
        tickers (List[str]): A list of ticker symbols.
        ticker_dfs (Dict[str, pd.DataFrame]): A dictionary where the keys are tickers and the values are DataFrames with historical data.
        window (int): The window size for calculating alpha and beta. By default is 60.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing a DataFrame with alpha and beta values, and a list of ticker symbols.
    """
    alpha = Alpha(
        insts=tickers,
        dfs=ticker_dfs,
        window=window,
    )
    data_path = os.path.join(os.getcwd(), "Data", f"alpha_{window}_rolling_dfs.obj")
    save_pickle(data_path, alpha.dfs)
