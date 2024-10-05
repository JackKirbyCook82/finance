# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Orders Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Contract
from support.mixins import Empty, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OrderCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class OrderVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Querys.PRODUCT: ["ticker", "expire", "strike"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}
    axes.update({Variables.Instruments.STOCK: list(map(str, Variables.Securities.Stocks)), Variables.Instruments.OPTION: list(map(str, Variables.Securities.Options))})
    data = {Variables.Datasets.EXPOSURE: ["quantity"]}

    def __init__(self, *args, **kwargs):
        self.options = list(map(str, self.axes[Variables.Instruments.OPTIONS]))
        self.stocks = list(map(str, self.axes[Variables.Instruments.STOCK]))
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.product = self.axes[Variables.Querys.PRODUCT]
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.identity = ["portfolio"]
        self.index = self.identity + self.product + self.security
        self.columns = self.data[Variables.Datasets.EXPOSURE]
        self.header = self.index + self.columns


class OrderCalculator(Sizing, Empty, Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = OrderVariables(*args, **kwargs)

    def __call__(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        for contract, dataframe in self.contracts(valuations):
            orders = self.execute(dataframe, *args, **kwargs)
            size = self.size(orders)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(orders): continue
            yield orders

    def contracts(self, valuations):
        assert isinstance(valuations, pd.DataFrame)
        for contract, dataframe in valuations.groupby(self.variables.contract):
            if self.empty(dataframe): continue
            yield Contract(*contract), dataframe

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        orders = self.calculate(valuations, *args, **kwargs)
        orders = pd.concat(orders, axis=0)
        orders = orders.reset_index(drop=True, inplace=False)
        return orders

    def calculate(self, valuations, *args, **kwargs):
        securities = self.securities(valuations, *args, **kwargs)
        orders = list(self.orders(securities, *args, **kwargs))
        return orders

    def securities(self, valuations, *args, **kwargs):
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in self.variables.stocks}
        options = valuations[self.variables.identity + self.variables.contract + self.variables.options + ["valuation", "strategy", "underlying"]]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        securities = pd.concat([options, stocks], axis=1)
        securities = securities[self.variables.identity + self.variables.contract + self.variables.options + self.variables.stocks]
        return securities

    def orders(self, securities, *args, **kwargs):
        for index, dataframe in securities.iterrows():
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
        stocks[self.variables.security + self.variables.columns] = stocks.apply(security, axis=1, result_type="expand")
        stocks = stocks[[column for column in stocks.columns if column != "security"]]
        contract = {key: value for key, value in securities[self.variables.identity + self.variables.contract].to_dict().items()}
        stocks = stocks.assign(**contract)
        return stocks

    def options(self, securities, *args, **kwargs):
        security = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        options = securities[self.variables.options].to_frame("strike")
        options = options.reset_index(names="security", drop=False, inplace=False)
        options[self.variables.security + self.variables.columns] = options.apply(security, axis=1, result_type="expand")
        options = options[[column for column in options.columns if column != "security"]]
        contract = {key: value for key, value in securities[self.variables.identity + self.variables.contract].to_dict().items()}
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

    @property
    def variables(self): return self.__variables



