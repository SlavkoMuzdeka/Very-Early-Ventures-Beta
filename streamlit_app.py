import os
import streamlit as st

from utils.alpha_utils import save_pickle
from utils.streamlit_utils import (
    create_alpha,
    get_alpha_beta_df,
    show_alpha_vs_beta,
    show_histogram_and_kde,
    show_alpha_beta_line_plots,
    get_tickers_and_ticker_dfs,
)

# --- Page configuration ---
st.set_page_config(
    page_title="Very Early Ventures - Beta Analysis", page_icon="📊", layout="wide"
)

st.title("📊 Very Early Ventures - Beta Analysis")


def calculate_and_store_alpha(tickers, ticker_dfs, window=None, use_rolling=True):
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
        tickers, ticker_dfs = get_tickers_and_ticker_dfs()

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
show_histogram_and_kde(
    dfs=[alpha_60_rolling_df, alpha_365_rolling_df, alpha_no_rolling_df],
    labels=["60 Days Rolling", "365 Days Rolling", "Full Period (no rolling)"],
    metric="Beta",
)
show_histogram_and_kde(
    dfs=[alpha_60_rolling_df, alpha_365_rolling_df, alpha_no_rolling_df],
    labels=["60 Days Rolling", "365 Days Rolling", "Full Period (no rolling)"],
    metric="Alpha",
    decimal_places=5,
)

symbols = st.session_state.tokens
symbols = [symbol for symbol in symbols if not symbol == "ETH-USD"]

st.divider()
token = st.selectbox("Select crypto asset", symbols)
show_alpha_beta_line_plots(token, dataset_mapping.get(selected_period))
