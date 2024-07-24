import pytz
import pandas as pd
import streamlit as st
import plotly.express as px

from alpha import Alpha
from datetime import datetime
from utils.alpha_utils import load_pickle
from data_models.YFinanceFetcher import YFinanceFetcher


def get_tickers_and_ticker_dfs():
    period_start = datetime(2023, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)

    obj_path = "yfinance_dataset.obj"
    data_fetcher = YFinanceFetcher()  # We are fetching data from Yahoo Finance

    tickers, ticker_dfs = data_fetcher.get_ticker_dfs(
        start=period_start, end=period_end, obj_path=obj_path
    )

    return tickers, ticker_dfs


def create_alpha(tickers, ticker_dfs, window=60, use_rolling=True):
    return Alpha(
        insts=tickers,
        dfs=ticker_dfs,
        window=window,
        use_rolling=use_rolling,
    )


def get_alpha_beta_df(alpha):
    beta_data = {}
    alpha_data = {}

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


def show_alpha_vs_beta(df):
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


def show_histogram_and_kde(dfs, labels, metric):
    st.divider()
    # Combine all DataFrames into one with an additional column to distinguish them
    combined_df = (
        pd.concat(dfs, keys=labels).reset_index(level=1, drop=True).reset_index()
    )
    combined_df.rename(columns={"index": "Dataset"}, inplace=True)

    fig = px.histogram(
        combined_df,
        x=metric,
        color="Dataset",
        marginal="violin",
        opacity=0.75,
        title=f"Distribution of {metric} Values",
        labels={metric: metric, "Dataset": "Dataset"},
    )

    fig.update_layout(
        xaxis_title=metric,
        yaxis_title="Frequency",
        title=f"Distribution of {metric} Values",
        height=650,  # Set the height of the plot
    )

    st.plotly_chart(fig, use_container_width=True)


def show_alpha_beta_line_plots(ticker, corr_to, dfs_path):
    dfs = load_pickle(dfs_path)

    alpha_col = f"alpha_{corr_to.lower()}"
    show_line_plot(
        x=dfs[ticker].index,
        y=dfs[ticker][alpha_col],
        title="Alpha",
    )

    beta_col = f"beta_{corr_to.lower()}"
    show_line_plot(
        x=dfs[ticker].index,
        y=dfs[ticker][beta_col],
        title="Beta",
    )


def show_line_plot(x, y, title):
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
