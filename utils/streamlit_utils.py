import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from typing import List
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from utils.yfinance_fetcher_utils import load_pickle


def show_alpha_vs_beta(df: pd.DataFrame):
    """
    Displays a scatter plot comparing Alpha vs Beta for all tokens.

    Args:
        df (pd.DataFrame): A DataFrame containing the Alpha and Beta values for each token.
    """
    hover_data = {"Token": df.index, "Alpha": True, "Beta": True}

    high_beta_positive_alpha = df[(df["Beta"] > 1) & (df["Alpha"] > 0)]
    top_5_tokens = high_beta_positive_alpha.nlargest(5, "Beta")

    df["Category"] = df.index.isin(top_5_tokens.index)
    df["Category"] = df["Category"].map({True: "Top 5 Tokens", False: "Other Tokens"})

    color_discrete_map = {
        "Top 5 Tokens": "#FF5733",
        "Other Tokens": "#3498DB",
    }

    fig = px.scatter(
        df,
        x="Alpha",
        y="Beta",
        title="Alpha vs Beta for all tokens",
        hover_data=hover_data,
        height=550,
        color="Category",
        color_discrete_map=color_discrete_map,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_kde(dfs: List[pd.DataFrame], labels: List[str], metrics: List[str]):
    """
    Displays KDE plots for multiple metrics across multiple DataFrames.

    Args:
        dfs (List[pd.DataFrame]): A list of DataFrames containing the metric values.
        labels (List[str]): A list of labels for each DataFrame.
        metrics (List[str]): A list of metrics to plot.
        decimal_places (int): The number of decimal places for statistical annotations.
    """
    colors = ["aqua", "mediumorchid", "lightcoral"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=metrics)
    decimal_places = 5
    for metric_idx, metric in enumerate(metrics):
        for index, (df, label) in enumerate(zip(dfs, labels)):
            # Compute KDE using scipy
            kde = gaussian_kde(df[metric].dropna())
            x_values = np.linspace(df[metric].min(), df[metric].max(), 1000)
            y_values = kde(x_values)

            # Compute statistics
            mean = df[metric].mean()
            median = df[metric].median()
            variance = df[metric].var()

            # Add the KDE line to the subplot with hover information
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{label} - {metric}",
                    line=dict(color=colors[index]),
                    hovertemplate=(
                        f"<b>{label} - {metric}</b><br>"
                        f"Value: %{{x}}<br>"
                        f"Density: %{{y}}<br>"
                        f"Mean: {mean:.{decimal_places}f}<br>"
                        f"Median: {median:.{decimal_places}f}<br>"
                        f"Variance: {variance:.{decimal_places}f}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=metric_idx + 1,
            )

    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability Density",
        height=550,
        title_text=f"Probability Density Distribution of Alpha and Beta",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


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
    columns = ["alpha_eth", "beta_eth"]
    axis_names = ["Alpha", "Beta"]

    for i, fig in enumerate(figures):
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=dfs[ticker].index,
                    y=dfs[ticker][columns[i]],
                    mode="lines",
                    name=f"{ticker} Alpha",
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
