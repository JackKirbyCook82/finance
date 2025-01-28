# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Partition

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ExposureCalculator(Sizing, Emptying, Partition, query=Querys.Settlement, title="Calculated"):
    header = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity"]

    def execute(self, holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        if self.empty(holdings): return
        for settlement, dataframe in self.partition(holdings):
            exposures = self.calculate(dataframe, *args, **kwargs)
            size = self.size(exposures)
            string = f"{str(settlement)}[{int(size):.0f}]"
            self.console(string)
            if self.empty(exposures): continue
            yield exposures

    def calculate(self, holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = self.securities(options, virtuals, *args, **kwargs)
        exposures = self.exposures(securities, *args, **kwargs)
        return exposures[self.header]

    @staticmethod
    def stocks(holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        stocks = holdings["instrument"] == Variables.Securities.Instrument.STOCK
        stocks = holdings.where(stocks).dropna(how="all", inplace=False)
        return stocks

    @staticmethod
    def options(holdings, *args, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        options = holdings["instrument"] == Variables.Securities.Instrument.OPTION
        options = holdings.where(options).dropna(how="all", inplace=False)
        puts = options["option"] == Variables.Securities.Option.PUT
        calls = options["option"] == Variables.Securities.Option.CALL
        options = options.where(puts | calls).dropna(how="all", inplace=False)
        return options

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
        if bool(stocks.empty): return pd.DataFrame()
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
        numerical = lambda position: 2 * int(bool(position is Variables.Securities.Position.LONG)) - 1
        enumerical = lambda value: Variables.Securities.Position.LONG if value > 0 else Variables.Securities.Position.SHORT
        holdings = lambda cols: cols["quantity"] * numerical(cols["position"])
        securities["quantity"] = securities.apply(holdings, axis=1)
        exposures = securities.groupby(index, as_index=False, sort=False).agg({"quantity": np.sum})
        exposures = exposures.where(exposures["quantity"] != 0).dropna(how="all", inplace=False)
        exposures["position"] = exposures["quantity"].apply(enumerical)
        exposures["quantity"] = exposures["quantity"].apply(np.abs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        return exposures




