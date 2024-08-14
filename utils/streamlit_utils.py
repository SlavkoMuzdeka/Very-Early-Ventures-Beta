import pytz
import pandas as pd
import streamlit as st
import plotly.express as px

from alpha import Alpha
from datetime import datetime
from typing import Tuple, List, Dict
from utils.alpha_utils import load_pickle
from data_models.YFinanceFetcher import YFinanceFetcher


def get_tickers_and_ticker_dfs() -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Fetches tickers and their corresponding historical data from Yahoo Finance.

    Returns:
        Tuple[List[str], Dict[str, pd.DataFrame]]: A tuple containing a list of tickers and a dictionary where the keys are tickers and the values are DataFrames with historical data.
    """
    period_start = datetime(2023, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)

    obj_path = "yfinance_dataset.obj"
    data_fetcher = YFinanceFetcher()  # We are fetching data from Yahoo Finance

    tickers, ticker_dfs = data_fetcher.get_ticker_dfs(
        start=period_start, end=period_end, obj_path=obj_path
    )

    return tickers, ticker_dfs


def create_alpha(
    tickers: List[str], ticker_dfs: Dict[str, pd.DataFrame], window=60, use_rolling=True
) -> Alpha:
    """
    Creates an Alpha object with the given tickers and historical data.

    Args:
        tickers (List[str]): A list of ticker symbols.
        ticker_dfs (Dict[str, pd.DataFrame]): A dictionary where the keys are tickers and the values are DataFrames with historical data.
        window (int): The window size for calculating alpha and beta.
        use_rolling (bool): Whether to use rolling calculations for alpha and beta.

    Returns:
        Alpha: An Alpha object initialized with the given parameters.
    """
    return Alpha(
        insts=tickers,
        dfs=ticker_dfs,
        window=window,
        use_rolling=use_rolling,
    )


def get_alpha_beta_df(alpha: Alpha):
    """
    Retrieves the alpha and beta values from an Alpha object and returns them in a DataFrame.

    Args:
        alpha (Alpha): The Alpha object containing alpha and beta data.

    Returns:
        pd.DataFrame: A DataFrame with the last values of beta and alpha for each token.
    """
    beta_data: Dict[str, pd.Series] = {}
    alpha_data: Dict[str, pd.Series] = {}

    for inst in alpha.insts:
        beta_data[inst] = alpha.dfs[inst]["beta_eth"]
        alpha_data[inst] = alpha.dfs[inst]["alpha_eth"]

    beta_df = pd.concat(beta_data, axis=1)
    beta_df.index = pd.to_datetime(beta_df.index)

    alpha_df = pd.concat(alpha_data, axis=1)
    alpha_df.index = pd.to_datetime(alpha_df.index)
    beta_last_values = beta_df.iloc[-1]
    alpha_last_values = alpha_df.iloc[-1]

    data = {
        token: {"Beta": beta_last_values[token], "Alpha": alpha_last_values[token]}
        for token in beta_last_values.index
    }

    return pd.DataFrame.from_dict(data, orient="index")


def show_alpha_vs_beta(df: pd.DataFrame):
    """
    Displays a scatter plot comparing Alpha vs Beta for all tokens.

    Args:
        df (pd.DataFrame): A DataFrame containing the Alpha and Beta values for each token.
    """
    hover_data = {"Token": df.index, "Alpha": True, "Beta": True}

    fig = px.scatter(
        df,
        x="Alpha",
        y="Beta",
        title="Alpha vs Beta for all tokens",
        hover_data=hover_data,
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_histogram_and_kde(
    dfs: List[pd.DataFrame], labels: List[str], metric: str, decimal_places=2
):
    """
    Displays histograms and KDE plots for a given metric across multiple DataFrames.

    Args:
        dfs (List[pd.DataFrame]): A list of DataFrames containing the metric values.
        labels (List[str]): A list of labels for each DataFrame.
        metric (str): The metric to plot.
        decimal_places (int): The number of decimal places for statistical annotations.
    """
    st.divider()

    colors = ["aqua", "mediumorchid", "lightcoral"]

    # Combine all DataFrames into one with an additional column to distinguish them
    combined_df = (
        pd.concat(dfs, keys=labels).reset_index(level=1, drop=True).reset_index()
    )
    combined_df.rename(columns={"index": "Dataset"}, inplace=True)

    fig = px.histogram(
        combined_df,
        x=metric,
        color="Dataset",
        opacity=0.75,
        histnorm="probability density",
        title=f"Probability Density Distribution of {metric} Values",
        labels={metric: metric, "Dataset": "Dataset"},
    )
    for i, trace in enumerate(fig.data):
        trace.update(marker_color=colors[i % len(colors)])

    # Add vertical lines for mean and median
    for index, df in enumerate(dfs):
        mean_value = df[metric].mean()
        median_value = df[metric].median()
        variance_value = df[metric].var()

        fig.add_vline(
            x=median_value,
            line_dash="dash",
            line_color=colors[index],
            annotation_text="",
            annotation_font_size=10,
        )

        # Add hover text with statistics
        fig.add_scatter(
            x=[median_value],
            y=[0],
            mode="markers",
            marker=dict(color=colors[index], symbol="x", size=10),
            text=[
                f"Median: {median_value:.{decimal_places}f}<br>Mean: {mean_value:.{decimal_places}f}<br>Variance: {variance_value:.{decimal_places}f}"
            ],
            hovertemplate=(
                f"Median: {median_value:.{decimal_places}f}<br>"
                f"Mean: {mean_value:.{decimal_places}f}<br>"
                f"Variance: {variance_value:.{decimal_places}f}<br>"
                f"<extra>Median Value</extra>"
            ),
            showlegend=False,
        )

    fig.update_layout(
        xaxis_title=metric,
        yaxis_title="Probability Density",
        title=f"Probability Density Distribution of {metric} Values",
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_alpha_beta_line_plots(ticker: str, dfs_path: str):
    """
    Displays line plots for Alpha and Beta values for a specific ticker.

    Args:
        ticker (str): The ticker symbol for which to plot Alpha and Beta.
        dfs_path (str): The path to the file containing the DataFrames with Alpha and Beta data.
    """
    dfs = load_pickle(dfs_path)

    show_line_plot(
        x=dfs[ticker].index,
        y=dfs[ticker]["alpha_eth"],
        title="Alpha",
    )

    show_line_plot(
        x=dfs[ticker].index,
        y=dfs[ticker]["beta_eth"],
        title="Beta",
    )


def show_line_plot(x: pd.DatetimeIndex, y: pd.Series, title: str):
    """
    Displays a line plot for a given x and y data.

    Args:
        x (pd.DatetimeIndex): The x-axis data (datetime values).
        y (pd.Series): The y-axis data (values to plot).
        title (str): The title of the plot.
    """
    fig = px.line(
        x=x,
        y=y,
        labels={"x": "Datetime", "y": f"{title}"},
        title=title,
        width=1200,
    )
    fig.update_traces(
        hovertemplate=f"<b>Date</b>: %{{x|%Y-%m-%d}}<br><b>Value</b>: %{{y}}<br>",
    )

    st.plotly_chart(fig, use_container_width=True)
