import os
import streamlit as st

from typing import List
from utils.streamlit_utils import (
    calculate_alpha,
    show_timeseries_plot,
    get_tickers_and_ticker_dfs,
)

# --- Page configuration ---
st.set_page_config(
    page_title="Very Early Ventures - Beta Analysis", page_icon="📊", layout="wide"
)

st.title("📊 Very Early Ventures - Beta Analysis")

if "created_alphas" not in st.session_state:
    with st.spinner("Calculating statistics..."):
        tickers, ticker_dfs = get_tickers_and_ticker_dfs()

        calculate_alpha(tickers, ticker_dfs)
        calculate_alpha(tickers, ticker_dfs, window=90)
        calculate_alpha(tickers, ticker_dfs, window=180)
        calculate_alpha(tickers, ticker_dfs, window=365)

        st.session_state.tickers = tickers
        st.session_state.created_alphas = True


selected_period = st.selectbox(
    "Select a period",
    ["60 Days Rolling", "90 Days Rolling", "180 Days Rolling", "365 Days Rolling"],
)

dataset_mapping = {
    "60 Days Rolling": os.path.join(os.getcwd(), "Data", "alpha_60_rolling_dfs.obj"),
    "90 Days Rolling": os.path.join(os.getcwd(), "Data", "alpha_90_rolling_dfs.obj"),
    "180 Days Rolling": os.path.join(os.getcwd(), "Data", "alpha_180_rolling_dfs.obj"),
    "365 Days Rolling": os.path.join(os.getcwd(), "Data", "alpha_365_rolling_dfs.obj"),
}

st.divider()
tokens: List[str] = st.session_state.tickers
tokens = sorted(tokens)
tokens = [token for token in tokens if not token == "ETH-USD"]

selected_tokens = st.multiselect(
    "Select crypto asset", default="AAVE-USD", options=tokens, max_selections=5
)
show_timeseries_plot(selected_tokens, dataset_mapping.get(selected_period))
