import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pyfinance.ols import PandasRollingOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


class Alpha:

    def __init__(self, insts, dfs, window, use_rolling=True):
        self.insts = insts
        self.dfs = dfs
        self.window = window
        self.use_rolling = use_rolling
        self.pre_compute()
        self.post_compute()
        self.logger = logging.getLogger(__name__)

    def pre_compute(self):
        for inst in self.insts:
            # Replace zero values to avoid log(0) warnings
            self.dfs[inst]["close"] = self.dfs[inst]["close"].replace(0, np.nan)
            self.dfs[inst] = self.dfs[inst].ffill().bfill()

            # log returns and volas
            self.dfs[inst]["log_ret"] = np.log(self.dfs[inst]["close"]) - np.log(
                self.dfs[inst]["close"].shift(1)
            )
            self.dfs[inst]["log_ret"] = self.dfs[inst]["log_ret"].bfill().ffill()
        return

    def _get_alpha_beta(self, inst, window, relative_to="ETH-USD"):
        x_other, y = self._format_x_other_and_y(inst, relative_to)

        while x_other.shape[0] <= window:
            window = int(window / 2)

        model_btc = PandasRollingOLS(y=y, x=x_other, window=window)

        return model_btc.alpha, model_btc.beta

    def _get_alpha_beta_no_rolling(self, inst, relative_to="ETH-USD"):
        x_other, y = self._format_x_other_and_y(inst, relative_to)

        # Fit OLS model
        X = sm.add_constant(x_other)
        model = sm.OLS(y, X).fit()

        return model.params.iloc[0], model.params.iloc[1]

    def _format_x_other_and_y(self, inst, relative_to):
        y = self.dfs[inst]["log_ret"]
        x_other = self.dfs[relative_to]["log_ret"]
        x_other = x_other.rename("log_ret_other")

        # for aligning on time merge and then take individual series again
        temp_merged = pd.concat([y, x_other], join="inner", axis=1)
        y = temp_merged["log_ret"]
        x_other = temp_merged["log_ret_other"]

        return x_other, y

    def post_compute(self):
        for inst in self.insts:
            # alpha & beta calculation to ETH
            if self.use_rolling:
                alpha, beta = self._get_alpha_beta(inst, self.window)
            else:
                alpha, beta = self._get_alpha_beta_no_rolling(inst)

            self.dfs[inst]["alpha_eth"] = alpha
            self.dfs[inst]["beta_eth"] = beta
        return
