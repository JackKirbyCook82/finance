# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime

from finance.variables import Variables
from support.pipelines import Processor
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureFiles", "ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


exposure_index = {"ticker": str, "instrument": int, "position": int, "strike": np.float32, "expire": np.datetime64}
exposure_parsers = {"instrument": lambda x: Variables.Instruments(int(x)), "position": lambda x: Variables.Positions(int(x))}
exposure_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])
exposure_columns = {"quantity": np.int32}


class ExposureFile(File, variable="exposure", datatype=pd.DataFrame, filename=exposure_filename, header=exposure_index | exposure_columns, parsers=exposure_parsers):
    pass


class ExposureCalculator(Processor):
    def execute(self, contents, *args, **kwargs):
        holdings = contents["holdings"]
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposures = self.holdings(securities, *args, *kwargs)
        exposures = self.expired(exposures, *args, **kwargs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        exposures = {"exposure": exposures}
        yield contents | exposures

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        stocks = dataframe["instrument"] == Variables.Instruments.STOCK
        stocks = dataframe.where(stocks).dropna(how="all", inplace=False)
        return stocks

    @staticmethod
    def options(dataframe, *args, **kwargs):
        puts = dataframe["instrument"] == Variables.Instruments.PUT
        calls = dataframe["instrument"] == Variables.Instruments.CALL
        options = dataframe.where(puts | calls).dropna(how="all", inplace=False)
        return options

    @staticmethod
    def virtuals(dataframe, *args, **kwargs):
        security = lambda instrument, position: dict(instrument=instrument, position=position)
        function = lambda records, instrument, position: pd.DataFrame.from_records([record | security(instrument, position) for record in records])
        stocklong = dataframe["position"] == Variables.Positions.LONG
        stocklong = dataframe.where(stocklong).dropna(how="all", inplace=False)
        stockshort = dataframe["position"] == Variables.Positions.SHORT
        stockshort = dataframe.where(stockshort).dropna(how="all", inplace=False)
        putlong = function(stockshort.to_dict("records"), Variables.Instruments.PUT, Variables.Positions.LONG)
        putshort = function(stocklong.to_dict("records"), Variables.Instruments.PUT, Variables.Positions.SHORT)
        calllong = function(stocklong.to_dict("records"), Variables.Instruments.CALL, Variables.Positions.LONG)
        callshort = function(stockshort.to_dict("records"), Variables.Instruments.CALL, Variables.Positions.SHORT)
        virtuals = pd.concat([putlong, putshort, calllong, callshort], axis=0)
        virtuals["strike"] = virtuals["strike"].apply(lambda strike: np.round(strike, decimals=2))
        return virtuals

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        index = [value for value in dataframe.columns if value not in ("position", "quantity")]
        numerical = lambda position: 2 * int(bool(position is Variables.Positions.LONG)) - 1
        enumerical = lambda value: Variables.Positions.LONG if value > 0 else Variables.Positions.SHORT
        holdings = lambda cols: cols["quantity"] * numerical(cols["position"])
        dataframe["quantity"] = dataframe.apply(holdings, axis=1)
        dataframe = dataframe.groupby(index, as_index=False).agg({"quantity": np.sum})
        dataframe = dataframe.where(dataframe["quantity"] != 0).dropna(how="all", inplace=False)
        dataframe["position"] = dataframe["quantity"].apply(enumerical)
        dataframe["quantity"] = dataframe["quantity"].apply(np.abs)
        return dataframe

    @staticmethod
    def expired(dataframe, *args, current, **kwargs):
        assert isinstance(current, Datetime)
        dataframe = dataframe.where(dataframe["expire"] >= current)
        dataframe = dataframe.dropna(how="all", inplace=False)
        return dataframe


class ExposureFiles(object):
    Exposure = ExposureFile



