# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Orders Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OrderCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class OrderCalculator(Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Logging.__init__(self, *args, **kwargs)
        Columns = ntuple("Columns", "stocks options")
        stocks = list(map(str, Variables.Securities.Stocks))
        options = list(map(str, Variables.Securities.Options))
        self.__columns = Columns(stocks, options)

    def execute(self, contract, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        orders = self.calculate(valuations, *args, **kwargs)
        size = self.size(orders)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(orders): return
        return orders

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.valuations(valuations, *args, **kwargs)
        orders = list(self.orders(valuations, *args, **kwargs))
        orders = pd.concat(orders, axis=0)
        orders = orders.reset_index(drop=True, inplace=False)
        return orders

    def valuations(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in self.columns.stocks}
        columns = list(Querys.Contract) + list(self.columns.options) + ["order", "valuation", "strategy", "underlying"]
        options = valuations[columns]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        valuations = pd.concat([options, stocks], axis=1)
        return valuations

    def orders(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        for index, series in securities.iterrows():
            stocks = self.securities(series, *args, columns=self.columns.stocks, **kwargs)
            options = self.securities(series, *args, columns=self.columns.options, **kwargs)
            virtuals = self.virtuals(stocks, *args, **kwargs)
            allocations = pd.concat([options, virtuals], axis=0).dropna(how="any", inplace=False)
            allocations = allocations.reset_index(drop=True, inplace=False)
            yield allocations

    @staticmethod
    def securities(securities, *args, columns, **kwargs):
        assert isinstance(securities, pd.Series)
        function = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        dataframe = securities[columns].to_frame("strike")
        dataframe = dataframe.reset_index(names="security", drop=False, inplace=False)
        security = list(Variables.Security) + ["quantity"]
        dataframe[security] = dataframe.apply(function, axis=1, result_type="expand")
        columns = [column for column in dataframe.columns if column != "security"]
        dataframe = dataframe[columns]
        contract = list(Querys.Contract)
        contract = {key: value for key, value in securities[contract].to_dict().items()}
        dataframe = dataframe.assign(**contract, order=securities["order"])
        return dataframe

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

    @property
    def columns(self): return self.__columns



