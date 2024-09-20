# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Contract
from support.processes import Calculator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HoldingVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}
    axes.update({Variables.Instruments.STOCK: list(Variables.Securities.Stocks), Variables.Instruments.OPTION: list(Variables.Securities.Options)})
    data = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}

    def __init__(self, *args, valuation, **kwargs):
        options = list(map(str, self.axes[Variables.Instruments.OPTIONS]))
        stocks = list(map(str, self.axes[Variables.Instruments.STOCK]))
        contract = self.axes[Variables.Querys.CONTRACT]
        security = self.axes[Variables.Datasets.SECURITY]
        self.stacking = self.data[valuation]
        self.header = list(contract) + list(security) + ["strike", "quantity"]
        self.securities = list(options) + list(stocks)
        self.options = list(options)
        self.stocks = list(stocks)
        self.valuation = valuation


class HoldingCalculator(Calculator, variables=HoldingVariables):
    def execute(self, contract, valuations, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(valuations, pd.DataFrame)
        valuations = self.unpivot(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        securities = pd.concat([valuations, stocks], axis=1)
        holdings = self.holdings(securities, *args, **kwargs)
        size = self.size(holdings)
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return valuations

    def unpivot(self, valuations, *args, **kwargs):
        index = set(valuations.columns) - ({"scenario"} | set(self.variables.stacking))
        valuations = valuations[list(index)].droplevel("scenario", axis=1)
        return valuations

    def stocks(self, valuations, *args, **kwargs):
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in list(self.variables.stocks)}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        return stocks

    def holdings(self, securities, *args, **kwargs):
        securities = securities[[column for column in list(self.variables.header) if column in securities.columns] + list(self.variables.securities)]
        contracts = [column for column in securities.columns if column not in list(self.variables.securities)]
        holdings = securities.melt(id_vars=contracts, value_vars=list(self.variables.securities), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[self.variables.header]




