# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Allocation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Contract
from support.processes import Calculator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AllocationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class AllocationVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}
    axes.update({Variables.Instruments.STOCK: list(Variables.Securities.Stocks), Variables.Instruments.OPTION: list(Variables.Securities.Options)})

    def __init__(self, *args, **kwargs):
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.options = list(map(str, self.axes[Variables.Instruments.OPTIONS]))
        self.stocks = list(map(str, self.axes[Variables.Instruments.STOCK]))


class AllocationCalculator(Calculator, variables=AllocationVariables):
    def execute(self, contract, valuations, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(valuations, pd.DataFrame)
        valuations.insert(0, "portfolio", range(1, 1 + len(valuations)))
        securities = self.securities(valuations, *args, **kwargs)
        allocations = list(self.allocations(securities, *args, **kwargs))
        allocations = pd.concat(allocations, axis=0)
        allocations = allocations.reset_index(drop=True, inplace=False)
        size = self.size(allocations)
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return allocations

    def securities(self, valuations, *args, **kwargs):
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in self.axes.stocks}
        options = valuations[self.variables.contract + self.variables.options + ["portfolio", "valuation", "strategy", "underlying"]]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        securities = pd.concat([options, stocks], axis=1)
        securities = securities[["portfolio"] + self.variables.contract + self.variables.options + self.variables.stocks]
        return securities

    def allocations(self, securities, *args, **kwargs):
        for portfolio, dataframe in securities.iterrows():
            stocks = self.stocks(dataframe, *args, **kwargs)
            options = self.options(dataframe, *args, **kwargs)
            virtuals = self.virtuals(stocks, *args, **kwargs)
            allocations = pd.concat([options, virtuals], axis=0).dropna(how="any", inplace=False)
            allocations = allocations.reset_index(drop=True, inplace=False)
            yield allocations

    def stocks(self, securities, *args, **kwargs):
        security = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        stocks = securities[self.variables.stocks].to_frame("strike")
        stocks = stocks.reset_index(names="security", drop=False, inplace=False)
        stocks[self.variables.security + ["quantity"]] = stocks.apply(security, axis=1, result_type="expand")
        stocks = stocks[[column for column in stocks.columns if column != "security"]]
        contract = {key: value for key, value in securities[["portfolio"] + self.variables.contract].to_dict().items()}
        stocks = stocks.assign(**contract)
        return stocks

    def options(self, securities, *args, **kwargs):
        security = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        options = securities[self.variables.options].to_frame("strike")
        options = options.reset_index(names="security", drop=False, inplace=False)
        options[self.variables.security + ["quantity"]] = options.apply(security, axis=1, result_type="expand")
        options = options[[column for column in options.columns if column != "security"]]
        contract = {key: value for key, value in securities[["portfolio"] + self.variables.contract].to_dict().items()}
        options = options.assign(**contract)
        return options

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
        stocklong = stocks["position"] == Variables.Positions.LONG
        stocklong = stocks.where(stocklong).dropna(how="all", inplace=False)
        stockshort = stocks["position"] == Variables.Positions.SHORT
        stockshort = stocks.where(stockshort).dropna(how="all", inplace=False)
        putlong = function(stockshort.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.PUT, Variables.Positions.LONG)
        putshort = function(stocklong.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.PUT, Variables.Positions.SHORT)
        calllong = function(stocklong.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.CALL, Variables.Positions.LONG)
        callshort = function(stockshort.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.CALL, Variables.Positions.SHORT)
        virtuals = pd.concat([putlong, putshort, calllong, callshort], axis=0)
        virtuals["strike"] = virtuals["strike"].apply(lambda strike: np.round(strike, decimals=2))
        return virtuals





