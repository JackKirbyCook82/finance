# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables
from support.pipelines import Processor
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureFiles", "ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


exposure_index = {"ticker": str, "expire": np.datetime64, "strike": np.float32, "instrument": int, "option": int, "position": int}
exposure_columns = {"quantity": np.int32}
exposure_parsers = {"expire": np.datetime64, "instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
exposure_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])


class ExposureFile(File, variable=Variables.Datasets.EXPOSURE, datatype=pd.DataFrame, filename=exposure_filename, header=exposure_index | exposure_columns, parsers=exposure_parsers):
    pass


class ExposureCalculator(Processor):
    def execute(self, contents, *args, **kwargs):
        holdings = contents[Variables.Datasets.HOLDINGS]
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposures = self.holdings(securities, *args, *kwargs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        exposures = {Variables.Datasets.EXPOSURE: exposures}
        yield contents | exposures

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        stocks = dataframe["instrument"] == Variables.Instruments.STOCK
        dataframe = dataframe.where(stocks).dropna(how="all", inplace=False)
        return dataframe

    @staticmethod
    def options(dataframe, *args, **kwargs):
        options = dataframe["instrument"] == Variables.Instruments.OPTION
        dataframe = dataframe.where(options).dropna(how="all", inplace=False)
        puts = dataframe["option"] == Variables.Options.PUT
        calls = dataframe["option"] == Variables.Options.CALL
        dataframe = dataframe.where(puts | calls).dropna(how="all", inplace=False)
        return dataframe

    @staticmethod
    def virtuals(dataframe, *args, **kwargs):
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
        if bool(dataframe.empty):
            return pd.DataFrame()
        stocklong = dataframe["position"] == Variables.Positions.LONG
        stocklong = dataframe.where(stocklong).dropna(how="all", inplace=False)
        stockshort = dataframe["position"] == Variables.Positions.SHORT
        stockshort = dataframe.where(stockshort).dropna(how="all", inplace=False)
        putlong = function(stockshort.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.PUT, Variables.Positions.LONG)
        putshort = function(stocklong.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.PUT, Variables.Positions.SHORT)
        calllong = function(stocklong.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.CALL, Variables.Positions.LONG)
        callshort = function(stockshort.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.CALL, Variables.Positions.SHORT)
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
        dataframe = dataframe.groupby(index, as_index=False, sort=False).agg({"quantity": np.sum})
        dataframe = dataframe.where(dataframe["quantity"] != 0).dropna(how="all", inplace=False)
        dataframe["position"] = dataframe["quantity"].apply(enumerical)
        dataframe["quantity"] = dataframe["quantity"].apply(np.abs)
        return dataframe


class ExposureFiles(object):
    Exposure = ExposureFile



