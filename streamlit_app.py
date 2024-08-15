import os
import pandas as pd
import streamlit as st

from typing import List, Dict, Tuple
from data_models.YFinanceFetcher import YFinanceFetcher
from utils.alpha_utils import create_alpha, get_alpha_beta_df
from utils.yfinance_fetcher_utils import save_pickle, get_tickers_and_ticker_dfs
from utils.streamlit_utils import (
    show_kde,
    show_alpha_vs_beta,
    show_timeseries_plot,
)

# --- Page configuration ---
st.set_page_config(
    page_title="Very Early Ventures - Beta Analysis", page_icon="📊", layout="wide"
)

st.title("📊 Very Early Ventures - Beta Analysis")


def calculate_and_store_alpha(
    tickers: List[str],
    ticker_dfs: Dict[str, pd.DataFrame],
    window=None,
    use_rolling=True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculates alpha and beta values for given tickers and stores the results.

    Args:
        tickers (List[str]): A list of ticker symbols.
        ticker_dfs (Dict[str, pd.DataFrame]): A dictionary where the keys are tickers and the values are DataFrames with historical data.
        window (Optional[int, None]): The window size for calculating alpha and beta. If None, no rolling is used.
        use_rolling (bool): Whether to use rolling calculations for alpha and beta.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing a DataFrame with alpha and beta values, and a list of ticker symbols.
    """
    alpha = create_alpha(tickers, ticker_dfs, window=window, use_rolling=use_rolling)
    df = get_alpha_beta_df(alpha)
    data_path = os.path.join(
        os.getcwd(), "Data", f"alpha_{window or 'no'}_rolling_dfs.obj"
    )
    save_pickle(data_path, alpha.dfs)
    return df, alpha.insts


if (
    "alpha_60_rolling_df" not in st.session_state
    or "alpha_365_rolling_df" not in st.session_state
    or "alpha_no_rolling_df" not in st.session_state
):
    with st.spinner("Calculating statistics..."):
        data_fetcher = YFinanceFetcher()
        tickers, ticker_dfs = get_tickers_and_ticker_dfs(data_fetcher)

        st.session_state.alpha_60_rolling_df, st.session_state.tokens = (
            calculate_and_store_alpha(tickers, ticker_dfs, window=60, use_rolling=True)
        )
        st.session_state.alpha_365_rolling_df, _ = calculate_and_store_alpha(
            tickers, ticker_dfs, window=365, use_rolling=True
        )
        st.session_state.alpha_no_rolling_df, _ = calculate_and_store_alpha(
            tickers, ticker_dfs, use_rolling=False
        )


selected_period = st.selectbox(
    "Select a period",
    ["60 Days Rolling", "365 Days Rolling", "Full Period (no rolling)"],
)

alpha_60_rolling_df = st.session_state.alpha_60_rolling_df
alpha_365_rolling_df = st.session_state.alpha_365_rolling_df
alpha_no_rolling_df = st.session_state.alpha_no_rolling_df

period_mapping_df = {
    "60 Days Rolling": alpha_60_rolling_df,
    "365 Days Rolling": alpha_365_rolling_df,
    "Full Period (no rolling)": alpha_no_rolling_df,
}

dataset_mapping = {
    "60 Days Rolling": os.path.join(os.getcwd(), "Data", "alpha_60_rolling_dfs.obj"),
    "365 Days Rolling": os.path.join(os.getcwd(), "Data", "alpha_365_rolling_dfs.obj"),
    "Full Period (no rolling)": os.path.join(
        os.getcwd(), "Data", "alpha_no_rolling_dfs.obj"
    ),
}

show_alpha_vs_beta(period_mapping_df.get(selected_period))

st.divider()
tokens: List[str] = st.session_state.tokens
tokens = [token for token in tokens if not token == "ETH-USD"]

selected_tokens = st.multiselect(
    "Select crypto asset", default="BTC-USD", options=tokens, max_selections=5
)
show_timeseries_plot(selected_tokens, dataset_mapping.get(selected_period))

st.divider()
st.subheader("KDE Analysis Across Multiple Time Periods")
show_kde(
    dfs=[alpha_60_rolling_df, alpha_365_rolling_df, alpha_no_rolling_df],
    labels=["60 Days Rolling", "365 Days Rolling", "Full Period (no rolling)"],
    metrics=["Beta", "Alpha"],
)
