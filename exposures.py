# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from functools import reduce

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging
from support.tables import Writer, Table, View, Header
from support.meta import ParametersMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureCalculator", "ExposureWriter", "ExposureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ExposureFormatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "instrument", "option", "position", "strike", "quantity"]
    formats = {"strike": lambda column: f"{column:.02f}", "quantity": lambda column: f"{column:.0f}"}


class ExposureView(View, ABC, datatype=pd.DataFrame, **dict(ExposureFormatting)): pass
class ExposureHeader(Header, ABC):
    def __init__(self, *args, **kwargs):
        index, columns = ["ticker", "expire", "instrument", "option", "position", "strike"], ["quantity"]
        super().__init__(*args, index=index, columns=columns, **kwargs)


class ExposureTable(Table, ABC, datatype=pd.DataFrame, viewtype=ExposureView, headertype=ExposureHeader):
    def obsolete(self, contract, *args, **kwargs):
        if not bool(self): return
        assert isinstance(contract, Querys.Contract)
        mask = [self[:, key] == value for key, value in contract.items()]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        self.remove(mask)


class ExposureCalculator(Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Logging.__init__(self, *args, **kwargs)

    def execute(self, contract, holdings, *args, **kwargs):
        if self.empty(holdings): return
        exposures = self.calculate(holdings, *args, **kwargs)
        size = self.size(exposures)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(exposures): return
        return exposures

    def calculate(self, holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = self.securities(options, virtuals, *args, **kwargs)
        exposures = self.exposures(securities, *args, *kwargs)
        return exposures

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


class ExposureWriter(Writer):
    def write(self, contract, exposures, *args, **kwargs):
        self.table.obsolete(contract, *args, **kwargs)
        if self.empty(exposures): return
        self.table.combine(exposures)
        self.table.reset()
        self.table.sort(["ticker", "expire"], reverse=[False, False])



