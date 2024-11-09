# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging
from support.meta import ParametersMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator", "HoldingFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HoldingParameters(metaclass=ParametersMeta):
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int, "strike": lambda strike: round(strike, 2)}
    types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
    dates = {"expire": "%Y%m%d"}

class HoldingFile(File, variable="holdings", datatype=pd.DataFrame, **dict(HoldingParameters)):
    @staticmethod
    def filename(*args, query, **kwargs):
        ticker = str(query.ticker).upper()
        expire = str(query.expire.strftime("%Y%m%d"))
        return "_".join([ticker, expire])

    @staticmethod
    def parameters(*args, filename, **kwargs):
        ticker, expire = str(filename).split("_")
        ticker = str(ticker).upper()
        expire = Datetime.strptime(expire, "%Y%m%d").date()
        return dict(ticker=ticker, expire=expire)


class HoldingCalculator(Logging, Sizing, Emptying):
    def __init__(self, *args, valuation, **kwargs):
        assert valuation in list(Variables.Valuations)
        Logging.__init__(self, *args, **kwargs)
        valuations = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
        self.__stacking = valuations[valuation]

    def execute(self, contract, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        holdings = self.calculate(valuations, *args, **kwargs)
        size = self.size(holdings)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(holdings): return
        return holdings

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.unpivot(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        dataframe = pd.concat([valuations, stocks], axis=1)
        holdings = self.holdings(dataframe, *args, **kwargs)
        return holdings

    def unpivot(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({"scenario"} | set(self.stacking))
        valuations = dataframe[list(index)].droplevel("scenario", axis=1)
        return valuations

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        header = list(Querys.Product) + list(Variables.Security) + ["quantity"]
        columns = [column for column in list(header) if column in dataframe.columns]
        securities = dataframe[columns + list(map(str, Variables.Securities))]
        holdings = securities.melt(id_vars=list(Querys.Contract), value_vars=list(map(str, Variables.Securities)), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[header]

    @staticmethod
    def stocks(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        stocks = list(map(str, Variables.Securities.Stocks))
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        return stocks

    @property
    def stacking(self): return self.__stacking



