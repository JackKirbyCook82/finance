# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Allocation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables
from finance.operations import Operations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AllocationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


allocation_stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}
allocation_options = list(map(str, Variables.Securities.Options))
allocation_stocks = list(map(str, Variables.Securities.Stocks))
allocation_contract = ["portfolio", "ticker", "expire"]
allocation_columns = ["strike", "instrument", "option", "position", "quantity"]
allocation_header = allocation_contract + allocation_columns


class AllocationCalculator(Operations.Processor):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        valuations = contents[self.valuation]
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.valuations(valuations, *args, **kwargs)
        valuations.insert(0, "portfolio", range(1, 1 + len(valuations)))
        securities = self.securities(valuations, *args, **kwargs)
        allocations = list(self.allocations(securities, *args, **kwargs))
        allocations = pd.concat(allocations, axis=0)
        allocations = allocations.reset_index(drop=True, inplace=False)
        allocations = {Variables.Datasets.ALLOCATION: allocations[allocation_header]}
        valuations = {self.valuation: valuations}
        yield contents | dict(allocations) | dict(valuations)

    def valuations(self, valuations, *args, **kwargs):
        columns = {column: np.NaN for column in allocation_options if column not in valuations.columns}
        for column, value in columns.items():
            valuations[column] = value
        index = set(valuations.columns) - ({"scenario"} | allocation_stacking[self.valuation])
        valuations = valuations.pivot(index=list(index), columns="scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        return valuations

    def allocations(self, securities, *args, **kwargs):
        for portfolio, dataframe in securities.iterrows():
            stocks = self.stocks(dataframe, *args, **kwargs)
            options = self.options(dataframe, *args, **kwargs)
            virtuals = self.virtuals(stocks, *args, **kwargs)
            dataframe = pd.concat([options, virtuals], axis=0).dropna(how="any", inplace=False)
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            yield dataframe

    @staticmethod
    def securities(valuations, *args, **kwargs):
        stocks = list(map(str, Variables.Securities.Stocks))
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        options = valuations[allocation_contract + allocation_options + ["valuation", "strategy", "underlying"]]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        securities = pd.concat([options, stocks], axis=1)
        securities = securities[allocation_contract + allocation_options + allocation_stocks]
        return securities

    @staticmethod
    def stocks(securities, *args, **kwargs):
        security = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        dataframe = securities[allocation_stocks].to_frame("strike")
        dataframe = dataframe.reset_index(names="security", drop=False, inplace=False)
        dataframe[["instrument", "option", "position", "quantity"]] = dataframe.apply(security, axis=1, result_type="expand")
        dataframe = dataframe[[column for column in dataframe.columns if column != "security"]]
        for key, value in securities[allocation_contract].to_dict().items():
            dataframe[key] = value
        return dataframe

    @staticmethod
    def options(securities, *args, **kwargs):
        security = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        dataframe = securities[allocation_options].to_frame("strike")
        dataframe = dataframe.reset_index(names="security", drop=False, inplace=False)
        dataframe[["instrument", "option", "position", "quantity"]] = dataframe.apply(security, axis=1, result_type="expand")
        dataframe = dataframe[[column for column in dataframe.columns if column != "security"]]
        for key, value in securities[allocation_contract].to_dict().items():
            dataframe[key] = value
        return dataframe

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
    def valuation(self): return self.__valuation



