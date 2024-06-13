# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Querys, Variables
from support.pipelines import Processor
from support.parsers import Header
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureFiles", "ExposureHeaders", "ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


exposure_index = {"ticker": str, "instrument": str, "position": str, "strike": np.float32, "expire": np.datetime64}
exposure_columns = {"quantity": np.int32}


class ExposureFile(File, variable="exposure", query=Querys.Contract, datatype=pd.DataFrame, header=exposure_index | exposure_columns): pass
class ExposureHeader(Header, variable="exposure", axes={"index": exposure_index, "columns": exposure_columns}): pass


class ExposureCalculator(Processor):
    def execute(self, contents, *args, **kwargs):
        holdings = contents["holdings"]
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposures = self.holdings(securities, *args, *kwargs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        exposures = {"exposure": exposures}
        yield contents | exposures

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        stocks = dataframe["instrument"] == str(Variables.Instruments.STOCK.name).lower()
        stocks = dataframe.where(stocks).dropna(how="all", inplace=False)
        return stocks

    @staticmethod
    def options(dataframe, *args, **kwargs):
        puts = dataframe["instrument"] == str(Variables.Instruments.PUT.name).lower()
        calls = dataframe["instrument"] == str(Variables.Instruments.CALL.name).lower()
        options = dataframe.where(puts | calls).dropna(how="all", inplace=False)
        return options

    @staticmethod
    def virtuals(dataframe, *args, **kwargs):
        security = lambda instrument, position: dict(instrument=str(instrument.name).lower(), position=str(position.name).lower())
        function = lambda records, instrument, position: pd.DataFrame.from_records([record | security(instrument, position) for record in records])
        stocklong = dataframe["position"] == str(Variables.Positions.LONG.name).lower()
        stocklong = dataframe.where(stocklong).dropna(how="all", inplace=False)
        stockshort = dataframe["position"] == str(Variables.Positions.SHORT.name).lower()
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
        numerical = lambda value: 2 * int(Variables.Positions[str(value).upper()] is Variables.Positions.LONG) - 1
        symbolical = lambda value: lambda cols: str(Variables.Positions.LONG.name).lower() if value > 0 else str(Variables.Positions.SHORT.name).lower()
        holdings = lambda cols: cols["quantity"] * numerical(cols["position"])
        dataframe["quantity"] = dataframe.apply(holdings)
        dataframe = dataframe.groupby(index, as_index=False).agg({"quantity": np.sum})
        dataframe = dataframe.where(dataframe["quantity"] != 0).dropna(how="all", inplace=False)
        dataframe["position"] = dataframe["quantity"].apply(symbolical)
        dataframe["quantity"] = dataframe["quantity"].apply(np.abs)
        return dataframe


class ExposureFiles(object):
    Exposure = ExposureFile

class ExposureHeaders(object):
    Exposure = ExposureHeader



