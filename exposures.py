# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Contract
from support.meta import ParametersMeta
from support.mixins import Sizing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class ExposureAxes(object, metaclass=ParametersMeta):
    security = ["instrument", "option", "position"]
    contract = ["ticker", "expire"]

    def __init__(self, *args, **kwargs):
        self.header = self.contract + self.security + ["strike", "quantity"]
        self.index = self.contract + self.security + ["strike"]
        self.columns = ["quantity"]


class ExposureCalculator(Sizing):
    def __init__(self, *args, **kwargs):
        self.__axes = ExposureAxes(*args, **kwargs)
        self.__logger = __logger__

    def calculate(self, contract, holdings, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(holdings, pd.DataFrame)
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposures = self.exposures(securities, *args, *kwargs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        size = self.size(exposures)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return exposures

    @staticmethod
    def stocks(holdings, *args, **kwargs):
        stocks = holdings["instrument"] == Variables.Instruments.STOCK
        stocks = holdings.where(stocks).dropna(how="all", inplace=False)
        return stocks

    @staticmethod
    def options(holdings, *args, **kwargs):
        options = holdings["instrument"] == Variables.Instruments.OPTION
        options = holdings.where(options).dropna(how="all", inplace=False)
        puts = options["option"] == Variables.Options.PUT
        calls = options["option"] == Variables.Options.CALL
        options = options.where(puts | calls).dropna(how="all", inplace=False)
        return options

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
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
    def exposures(securities, *args, **kwargs):
        index = [value for value in securities.columns if value not in ("position", "quantity")]
        numerical = lambda position: 2 * int(bool(position is Variables.Positions.LONG)) - 1
        enumerical = lambda value: Variables.Positions.LONG if value > 0 else Variables.Positions.SHORT
        holdings = lambda cols: cols["quantity"] * numerical(cols["position"])
        securities["quantity"] = securities.apply(holdings, axis=1)
        exposures = securities.groupby(index, as_index=False, sort=False).agg({"quantity": np.sum})
        exposures = exposures.where(exposures["quantity"] != 0).dropna(how="all", inplace=False)
        exposures["position"] = exposures["quantity"].apply(enumerical)
        exposures["quantity"] = exposures["quantity"].apply(np.abs)
        return exposures

    @property
    def logger(self): return self.__logger
    @property
    def axes(self): return self.__axes



