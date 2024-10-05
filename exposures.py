# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC
from functools import reduce

from finance.variables import Variables, Contract
from support.mixins import Empty, Sizing, Logging
from support.tables import Table, View
from support.meta import ParametersMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class ExposureFormatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "instrument", "option", "position", "strike", "quantity"]
    formats = {"strike": lambda column: f"{column:.02f}", "quantity": lambda column: f"{column:.0f}"}


class ExposureVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Querys.PRODUCT: ["ticker", "expire", "strike"], Variables.Querys.SECURITY: ["instrument", "option", "position"]}
    data = {Variables.Datasets.EXPOSURE: ["quantity"]}

    def __init__(self, *args, **kwargs):
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.product = self.axes[Variables.Querys.PRODUCT]
        self.security = self.axes[Variables.Querys.SECURITY]
        self.index = self.product + self.security
        self.columns = self.data[Variables.Datasets.EXPOSURE]
        self.header = self.index + self.columns


class ExposureView(View, ABC, datatype=pd.DataFrame, **dict(ExposureFormatting)): pass
class ExposureTable(Table, ABC, datatype=pd.DataFrame, view=ExposureView, variable=Variables.Datasets.EXPOSURE): pass


class ExposureCalculator(Sizing, Empty, Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = ExposureVariables(*args, **kwargs)

    def __call__(self, holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        for contract, dataframe in self.contracts(holdings):
            exposures = self.execute(holdings, *args, **kwargs)
            size = self.size(exposures)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(exposures): continue
            yield exposures

    def contracts(self, holdings):
        assert isinstance(holdings, pd.DataFrame)
        for contract, dataframe in holdings.groupby(self.variables.contract):
            if self.empty(dataframe): continue
            yield Contract(*contract), dataframe

    def execute(self, holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        exposures = self.calculate(holdings, *args, **kwargs)
        return exposures

    def calculate(self, holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = self.securities(options, virtuals, *args, **kwargs)
        exposures = self.exposures(securities, *args, *kwargs)
        yield exposures

    @staticmethod
    def stocks(holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        stocks = holdings["instrument"] == Variables.Instruments.STOCK
        stocks = holdings.where(stocks).dropna(how="all", inplace=False)
        return stocks

    @staticmethod
    def options(holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        options = holdings["instrument"] == Variables.Instruments.OPTION
        options = holdings.where(options).dropna(how="all", inplace=False)
        puts = options["option"] == Variables.Options.PUT
        calls = options["option"] == Variables.Options.CALL
        options = options.where(puts | calls).dropna(how="all", inplace=False)
        return options

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
        if bool(stocks.empty): return pd.DataFrame()
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

    @staticmethod
    def securities(options, virtuals, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(virtuals, pd.DataFrame)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        return securities

    @staticmethod
    def exposures(securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        index = [value for value in securities.columns if value not in ("position", "quantity")]
        numerical = lambda position: 2 * int(bool(position is Variables.Positions.LONG)) - 1
        enumerical = lambda value: Variables.Positions.LONG if value > 0 else Variables.Positions.SHORT
        holdings = lambda cols: cols["quantity"] * numerical(cols["position"])
        securities["quantity"] = securities.apply(holdings, axis=1)
        exposures = securities.groupby(index, as_index=False, sort=False).agg({"quantity": np.sum})
        exposures = exposures.where(exposures["quantity"] != 0).dropna(how="all", inplace=False)
        exposures["position"] = exposures["quantity"].apply(enumerical)
        exposures["quantity"] = exposures["quantity"].apply(np.abs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        return exposures

    @property
    def variables(self): return self.__variables


class ExposureWriter(Sizing, Empty, Logging):
    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = ExposureVariables(*args, **kwargs)
        self.__table = table

    def __call__(self, exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        for contract, dataframe in self.contracts(exposures):
            with self.table.mutex:
                self.obsolete(contract, *args, **kwargs)
                self.execute(dataframe, *args, **kwargs)

    def contracts(self, exposures):
        assert isinstance(exposures, pd.DataFrame)
        for contract, dataframe in exposures.groupby(self.variables.contract):
            if self.empty(dataframe): continue
            yield Contract(*contract), dataframe

    def obsolete(self, contract, *args, **kwargs):
        contract = lambda table: [table[key] == value for key, value in zip(self.variables.contract, contract)]
        obsolete = lambda table: reduce(lambda x, y: x & y, contract(table))
        self.table.remove(obsolete)

    def execute(self, exposures, *args, **kwargs):
        sorting = dict(ticker=False, expire=False)
        self.table.combine(exposures)
        self.table.reset()
        self.table.sort(list(sorting.keys()), reverse=list(sorting.values()))

    @property
    def variables(self): return self.__variables
    @property
    def table(self): return self.__table



