# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Securities Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from finance.variables import Querys
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, stocks, options, technicals, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        if self.empty(options): return
        stocks = self.stocks(stocks, technicals)
        options = self.options(options, stocks)
        querys = self.groups(options, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        size = self.size(options)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(options): return
        yield options

    @staticmethod
    def stocks(stocks, technicals, *args, **kwargs):
        technicals = technicals.where(technicals["date"] == technicals["date"].max()).dropna(how="all", inplace=False)
        stocks = stocks.merge(technicals[["ticker", "trend", "volatility"]], how="left", on=list(Querys.Symbol), sort=False, suffixes=("", "_"))
        return stocks

    @staticmethod
    def options(options, stocks, *args, **kwargs):
        underlying = lambda dataframe: np.average(dataframe["price"], weights=dataframe["size"])
        volatility = lambda dataframe: np.average(dataframe["volatility"])
        trend = lambda dataframe: np.average(dataframe["trend"])
        stocks = [{"ticker": str(ticker), "underlying": underlying(dataframe), "trend": trend(dataframe), "volatility": volatility(dataframe)} for ticker, dataframe in stocks.groupby("ticker")]
        stocks = pd.DataFrame.from_records(stocks)
        options = options.merge(stocks, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", "_"))
        return options


