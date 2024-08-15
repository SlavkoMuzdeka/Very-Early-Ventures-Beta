import pandas as pd

from alpha import Alpha
from typing import List, Dict


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
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.dropna()

    return df
