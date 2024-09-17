# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from itertools import product

from finance.variables import Variables, Contract
from support.meta import ParametersMeta
from support.mixins import Sizing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HoldingAxes(object, metaclass=ParametersMeta):
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    securities = list(map(str, Variables.Securities))
    scenarios = list(Variables.Scenarios)
    security = ["instrument", "option", "position"]
    arbitrage = ["apy", "npv", "cost"]
    contract = ["ticker", "expire"]

    def __init__(self, *args, valuation, **kwargs):
        valuation = str(valuation).lower()
        self.stacking = list(product(getattr(self, valuation), self.scenarios))
        self.header = self.contract + self.security + ["strike", "quantity"]
        self.index = self.contract + self.security + ["strike"]
        self.columns = ["quantity"]


class HoldingCalculator(Sizing):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__axes = HoldingAxes(*args, valuation=valuation, **kwargs)
        self.__logger = __logger__

    def calculate(self, contract, valuations, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(valuations, pd.DataFrame)
        options = self.options(valuations, *args, **kwargs)
        stocks = self.stocks(options, *args, **kwargs)
        securities = pd.concat([options, stocks], axis=1)
        holdings = self.holdings(securities, *args, **kwargs)
        size = self.size(holdings)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return valuations

    def options(self, valuations, *args, **kwargs):
        stacking = [column[0] for column in self.axes.stacking]
        index = set(valuations.columns) - ({"scenario"} | set(stacking))
        options = valuations[list(index)].droplevel("scenario", axis=1)
        return options

    def stocks(self, options, *args, **kwargs):
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in list(self.axes.stocks)}
        stocks = options.apply(function, axis=1, result_type="expand")
        return stocks

    def holdings(self, securities, *args, **kwargs):
        securities = securities[[column for column in list(self.axes.header) if column in securities.columns] + list(self.axes.securities)]
        contracts = [column for column in securities.columns if column not in list(self.axes.securities)]
        holdings = securities.melt(id_vars=contracts, value_vars=list(self.axes.securities), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[list(self.axes.header)]

    @property
    def logger(self): return self.__logger
    @property
    def axes(self): return self.__axes



