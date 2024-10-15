# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Orders Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Pipelining, Sourcing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OrderCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class OrderCalculator(Pipelining, Sourcing, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        for contract, dataframe in self.source(valuations, keys=list(Querys.Contract)):
            contract = Querys.Contract(contract)
            if self.empty(dataframe): continue
            orders = self.calculate(dataframe, *args, **kwargs)
            size = self.size(orders)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(orders): continue
            yield orders

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        securities = self.securities(valuations, *args, **kwargs)
        orders = list(self.orders(securities, *args, **kwargs))
        orders = pd.concat(orders, axis=0)
        orders = orders.reset_index(drop=True, inplace=False)
        return orders

    def orders(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        for index, dataframe in securities.iterrows():
            stocks = self.stocks(dataframe, *args, **kwargs)
            options = self.options(dataframe, *args, **kwargs)
            virtuals = self.virtuals(stocks, *args, **kwargs)
            allocations = pd.concat([options, virtuals], axis=0).dropna(how="any", inplace=False)
            allocations = allocations.reset_index(drop=True, inplace=False)
            yield allocations

    @staticmethod
    def securities(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        stocks, options = list(map(str, Variables.Securities.Stocks)), list(map(str, Variables.Securities.Options))
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        columns = list(Querys.Contract) + list(options) + ["valuation", "strategy", "underlying"]
        options = valuations[columns]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        securities = pd.concat([options, stocks], axis=1)
        columns = list(Querys.Contract) + list(options) + list(stocks)
        securities = securities[columns]
        return securities

    @staticmethod
    def stocks(securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        function = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        stocks = list(map(str, Variables.Securities.Stocks))
        stocks = securities[stocks].to_frame("strike")
        stocks = stocks.reset_index(names="security", drop=False, inplace=False)
        security = list(Variables.Security) + ["quantity"]
        stocks[security] = stocks.apply(function, axis=1, result_type="expand")
        columns = [column for column in stocks.columns if column != "security"]
        stocks = stocks[columns]
        contract = list(Variables.Contract)
        contract = {key: value for key, value in securities[contract].to_dict().items()}
        stocks = stocks.assign(**contract)
        return stocks

    @staticmethod
    def options(securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        function = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        options = list(map(str, Variables.Securities.Options))
        options = securities[options].to_frame("strike")
        options = options.reset_index(names="security", drop=False, inplace=False)
        security = list(Variables.Security) + ["quantity"]
        options[security] = options.apply(function, axis=1, result_type="expand")
        columns = [column for column in options.columns if column != "security"]
        options = options[columns]
        contract = list(Variables.Contract)
        contract = {key: value for key, value in securities[contract].to_dict().items()}
        options = options.assign(**contract)
        return options

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
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



