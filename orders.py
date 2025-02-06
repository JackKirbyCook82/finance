# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Orders Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OrderCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class OrderCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    header = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity", "order"]

    def execute(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        if self.empty(prospects): return
        for settlement, dataframe in self.partition(prospects, by=Querys.Settlement):
            contents = self.prospects(dataframe, *args, **kwargs)
            orders = self.calculate(contents, *args, **kwargs)
            size = self.size(orders)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(orders): continue
            yield orders

    def calculate(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        orders = list(self.calculator(prospects, *args, **kwargs))
        orders = pd.concat(orders, axis=0)
        orders = orders.reset_index(drop=True, inplace=False)
        return orders[self.header]

    def calculator(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        for index, series in prospects.iterrows():
            stocks = self.securities(series, *args, columns=list(map(str, Securities.Stocks)), **kwargs)
            options = self.securities(series, *args, columns=list(map(str, Securities.Options)), **kwargs)
            virtuals = self.virtuals(stocks, *args, **kwargs)
            allocations = pd.concat([options, virtuals], axis=0).dropna(how="any", inplace=False)
            allocations = allocations.reset_index(drop=True, inplace=False)
            yield allocations

    @staticmethod
    def prospects(prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in list(map(str, Securities.Stocks))}
        options = prospects[list(Querys.Settlement) + list(map(str, Securities.Options)) + ["order", "valuation", "strategy", "underlying"]]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        prospects = pd.concat([options, stocks], axis=1)
        return prospects

    @staticmethod
    def securities(prospects, *args, columns, **kwargs):
        assert isinstance(prospects, pd.Series)
        function = lambda cols: list(Securities(cols["security"])) + [1]
        dataframe = prospects[columns].to_frame("strike")
        dataframe = dataframe.reset_index(names="security", drop=False, inplace=False)
        dataframe[list(Variables.Securities.Security) + ["quantity"]] = dataframe.apply(function, axis=1, result_type="expand")
        columns = [column for column in dataframe.columns if column != "security"]
        dataframe = dataframe[columns]
        contract = {key: value for key, value in prospects[list(Querys.Settlement)].to_dict().items()}
        dataframe = dataframe.assign(**contract, order=prospects["order"])
        return dataframe

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
        stocklong = stocks["position"] == Variables.Securities.Position.LONG
        stocklong = stocks.where(stocklong).dropna(how="all", inplace=False)
        stockshort = stocks["position"] == Variables.Securities.Position.SHORT
        stockshort = stocks.where(stockshort).dropna(how="all", inplace=False)
        putlong = function(stockshort.to_dict("records"), Variables.Securities.Instrument.OPTION, Variables.Securities.Option.PUT, Variables.Securities.Position.LONG)
        putshort = function(stocklong.to_dict("records"), Variables.Securities.Instrument.OPTION, Variables.Securities.Option.PUT, Variables.Securities.Position.SHORT)
        calllong = function(stocklong.to_dict("records"), Variables.Securities.Instrument.OPTION, Variables.Securities.Option.CALL, Variables.Securities.Position.LONG)
        callshort = function(stockshort.to_dict("records"), Variables.Securities.Instrument.OPTION, Variables.Securities.Option.CALL, Variables.Securities.Position.SHORT)
        virtuals = pd.concat([putlong, putshort, calllong, callshort], axis=0)
        virtuals["strike"] = virtuals["strike"].apply(lambda strike: np.round(strike, decimals=2))
        return virtuals


