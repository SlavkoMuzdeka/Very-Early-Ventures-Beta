import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Tuple, List, Dict
from pyfinance.ols import PandasRollingOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


class Alpha:

    def __init__(
        self,
        insts: List[str],
        dfs: Dict[str, pd.DataFrame],
        window: int,
        use_rolling=True,
    ):
        """
        Initializes the Alpha class with given instruments, dataframes, and calculation settings.

        Args:
            insts (list): A list of instrument identifiers (e.g., cryptocurrency symbols).
            dfs (Dict[str, DataFrame]): A dictionary of pandas DataFrames with historical data for each instrument.
            window (int): The rolling window size for calculating alpha and beta.
            use_rolling (bool): Flag to indicate whether to use rolling calculations (default is True).
        """
        self.insts = insts
        self.dfs = dfs
        self.window = window
        self.use_rolling = use_rolling
        self.pre_compute()
        self.post_compute()
        self.logger = logging.getLogger(__name__)

    def pre_compute(self):
        """
        Prepares the data by filling missing values and calculating log returns.

        This method processes each instrument's data, replacing zero values and calculating log returns to ensure data consistency.
        """
        for inst in self.insts:
            # Replace zero values to avoid log(0) warnings
            self.dfs[inst]["close"] = self.dfs[inst]["close"].replace(0, np.nan)
            self.dfs[inst] = self.dfs[inst].ffill().bfill()

            # Calculate log returns
            self.dfs[inst]["log_ret"] = np.log(self.dfs[inst]["close"]) - np.log(
                self.dfs[inst]["close"].shift(1)
            )
            self.dfs[inst]["log_ret"] = self.dfs[inst]["log_ret"].bfill().ffill()
        return

    def _get_alpha_beta(
        self, inst: str, window: int, relative_to="ETH-USD"
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates the alpha and beta using a rolling OLS model for a given instrument relative to another.

        Args:
            inst (str): The instrument for which alpha and beta are calculated.
            window (int): The rolling window size.
            relative_to (str): The instrument relative to which alpha and beta are calculated (default is "ETH-USD").

        Returns:
            tuple: A tuple containing the alpha and beta series.
        """
        x_other, y = self._format_x_other_and_y(inst, relative_to)

        while x_other.shape[0] <= window:
            window = int(window / 2)

        model_btc = PandasRollingOLS(y=y, x=x_other, window=window)

        return model_btc.alpha, model_btc.beta

    def _get_alpha_beta_no_rolling(
        self, inst: str, relative_to="ETH-USD"
    ) -> Tuple[float, float]:
        """
        Calculates the alpha and beta without rolling for a given instrument relative to another.

        Args:
            inst (str): The instrument for which alpha and beta are calculated.
            relative_to (str): The instrument relative to which alpha and beta are calculated (default is "ETH-USD").

        Returns:
            tuple: A tuple containing the alpha and beta values.
        """
        x_other, y = self._format_x_other_and_y(inst, relative_to)

        # Fit OLS model
        X = sm.add_constant(x_other)
        model = sm.OLS(y, X).fit()

        return model.params.iloc[0], model.params.iloc[1]

    def _format_x_other_and_y(
        self, inst: str, relative_to: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Formats and aligns the log returns for the target instrument and the reference instrument.

        Args:
            inst (str): The target instrument.
            relative_to (str): The reference instrument.

        Returns:
            tuple: A tuple containing the aligned log returns of the reference and target instruments.
        """
        y = self.dfs[inst]["log_ret"]
        x_other = self.dfs[relative_to]["log_ret"]
        x_other = x_other.rename("log_ret_other")

        # Align on time and merge
        temp_merged = pd.concat([y, x_other], join="inner", axis=1)
        y = temp_merged["log_ret"]
        x_other = temp_merged["log_ret_other"]

        return x_other, y

    def post_compute(self):
        """
        Computes alpha and beta for all instruments and stores the results in the dataframes.

        This method calculates alpha and beta for each instrument either using rolling windows
        or without rolling, depending on the initialization settings.
        """
        for inst in self.insts:
            # Calculate alpha & beta relative to ETH
            if self.use_rolling:
                alpha, beta = self._get_alpha_beta(inst, self.window)
            else:
                alpha, beta = self._get_alpha_beta_no_rolling(inst)

            self.dfs[inst]["alpha_eth"] = alpha
            self.dfs[inst]["beta_eth"] = beta
        return
