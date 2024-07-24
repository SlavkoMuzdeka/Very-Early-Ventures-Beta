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

if (
    "alpha_60_rolling_df" not in st.session_state
    or "alpha_365_rolling_df" not in st.session_state
    or "alpha_no_rolling_df" not in st.session_state
):
    with st.spinner("Calculating statistics..."):
        tickers, ticker_dfs = get_tickers_and_ticker_dfs()

        alpha_60 = create_alpha(tickers, ticker_dfs, window=60, use_rolling=True)
        df = get_alpha_beta_df(alpha_60)
        st.session_state.alpha_60_rolling_df = df
        st.session_state.tokens = alpha_60.insts
        save_pickle("Data/alpha_60_dfs.obj", alpha_60.dfs)

        alpha_365 = create_alpha(tickers, ticker_dfs, window=365, use_rolling=True)
        df = get_alpha_beta_df(alpha_365)
        st.session_state.alpha_365_rolling_df = df
        save_pickle("Data/alpha_365_dfs.obj", alpha_365.dfs)

        alpha_no_rolling = create_alpha(tickers, ticker_dfs, use_rolling=False)
        df = get_alpha_beta_df(alpha_no_rolling)
        st.session_state.alpha_no_rolling_df = df
        save_pickle("Data/alpha_no_rolling_dfs.obj", alpha_no_rolling.dfs)


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
    "60 Days Rolling": "Data/alpha_60_dfs.obj",
    "365 Days Rolling": "Data/alpha_365_dfs.obj",
    "Full Period (no rolling)": "Data/alpha_no_rolling_dfs.obj",
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
)

symbols = st.session_state.tokens

corr_to = st.radio("In correlation with?", ["BTC", "ETH"], horizontal=True)

if corr_to == "BTC":
    symbols = [symbol for symbol in symbols if not symbol == "BTC-USD"]
elif corr_to == "ETH":
    symbols = [symbol for symbol in symbols if not symbol == "ETH-USD"]

token = st.selectbox("Select crypto asset", symbols)


show_alpha_beta_line_plots(token, corr_to, dataset_mapping.get(selected_period))
