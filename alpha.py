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

            inst_log_vol = (
                np.log(self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1))
                .rolling(30)
                .std()
            )
            self.dfs[inst]["log_vol"] = inst_log_vol
            self.dfs[inst]["log_vol"] = self.dfs[inst]["log_vol"].ffill().fillna(0)
            self.dfs[inst]["log_vol"] = np.where(
                self.dfs[inst]["log_vol"] < 0.005, 0.005, self.dfs[inst]["log_vol"]
            )

            # 7 day returns
            self.dfs[inst]["ret_7_day"] = -1 + self.dfs[inst]["close"] / self.dfs[inst][
                "close"
            ].shift(7)

            # 30 day returns
            self.dfs[inst]["ret_30_day"] = -1 + self.dfs[inst]["close"] / self.dfs[
                inst
            ]["close"].shift(30)

        return

    def _get_alpha_beta(self, inst, window, relative_to="BTC-USD", updays=0):
        y = self.dfs[inst]["log_ret"]
        x_other = self.dfs[relative_to]["log_ret"]
        x_other = x_other.rename("log_ret_other")

        if updays != 0:
            x_other = x_other.loc[(updays * x_other) > 0]

        # for aligning on time merge and then take individual series again
        temp_merged = pd.concat([y, x_other], join="inner", axis=1)
        y = temp_merged["log_ret"]
        x_other = temp_merged["log_ret_other"]

        while x_other.shape[0] <= window:
            window = int(window / 2)

        model_btc = PandasRollingOLS(y=y, x=x_other, window=window)

        return model_btc.alpha, model_btc.beta

    def _get_alpha_beta_no_rolling(self, inst, relative_to="BTC-USD"):
        y = self.dfs[inst]["log_ret"]
        x_other = self.dfs[relative_to]["log_ret"]
        x_other = x_other.rename("log_ret_other")

        temp_merged = pd.concat([y, x_other], join="inner", axis=1)
        y = temp_merged["log_ret"]
        x_other = temp_merged["log_ret_other"]

        # Fit OLS model
        X = sm.add_constant(x_other)
        model = sm.OLS(y, X).fit()

        return model.params.iloc[0], model.params.iloc[1]

    def post_compute(self):
        for inst in self.insts:
            # alpha beta calculation to BTC
            if self.use_rolling:
                alpha, beta = self._get_alpha_beta(inst, self.window)
            else:
                alpha, beta = self._get_alpha_beta_no_rolling(inst)

            # alpha factors simple
            self.dfs[inst]["alpha_btc"] = alpha

            # beta to btc and eth
            self.dfs[inst]["beta_btc"] = beta

            # alpha beta calculation to ETH
            if self.use_rolling:
                alpha, beta = self._get_alpha_beta(
                    inst, self.window, relative_to="ETH-USD"
                )
            else:
                alpha, beta = self._get_alpha_beta_no_rolling(
                    inst, relative_to="ETH-USD"
                )

            # alpha factors simple
            self.dfs[inst]["alpha_eth"] = alpha

            # beta to btc and eth
            self.dfs[inst]["beta_eth"] = beta
        return
