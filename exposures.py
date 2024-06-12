# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Contract, Instruments, Positions
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
exposure_columns = {"entry": np.datetime64, "quantity": np.int32}


class ExposureFile(File, variable="exposure", query=Contract, datatype=pd.DataFrame, header=exposure_index | exposure_columns): pass
class ExposureHeader(Header, variable="exposure", axes={"index": exposure_index, "columns": exposure_columns}): pass


class ExposureCalculator(Processor):
    def execute(self, contents, *args, **kwargs):
        holdings = contents["holdings"]
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposure = self.holdings(securities, *args, *kwargs)
        exposure = exposure.reset_index(drop=True, inplace=False)
        yield contents | dict(exposure=exposure)

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        stocks = dataframe["instrument"] == str(Instruments.STOCK.name).lower()
        stocks = dataframe.where(stocks).dropna(how="all", inplace=False)
        return stocks

    @staticmethod
    def options(dataframe, *args, **kwargs):
        puts = dataframe["instrument"] == str(Instruments.PUT.name).lower()
        calls = dataframe["instrument"] == str(Instruments.CALL.name).lower()
        options = dataframe.where(puts | calls).dropna(how="all", inplace=False)
        return options

    @staticmethod
    def virtuals(dataframe, *args, **kwargs):
        security = lambda instrument, position: dict(instrument=str(instrument.name).lower(), position=str(position.name).lower())
        function = lambda records, instrument, position: pd.DataFrame.from_records([record | security(instrument, position) for record in records])
        stocklong = dataframe["position"] == str(Positions.LONG.name).lower()
        stocklong = dataframe.where(stocklong).dropna(how="all", inplace=False)
        stockshort = dataframe["position"] == str(Positions.SHORT.name).lower()
        stockshort = dataframe.where(stockshort).dropna(how="all", inplace=False)
        putlong = function(stockshort.to_dict("records"), Instruments.PUT, Positions.LONG)
        putshort = function(stocklong.to_dict("records"), Instruments.PUT, Positions.SHORT)
        calllong = function(stocklong.to_dict("records"), Instruments.CALL, Positions.LONG)
        callshort = function(stockshort.to_dict("records"), Instruments.CALL, Positions.SHORT)
        virtuals = pd.concat([putlong, putshort, calllong, callshort], axis=0)
        virtuals["strike"] = virtuals["strike"].apply(lambda strike: np.round(strike, decimals=2))
        return virtuals

    @staticmethod
    def holdings(dataframe, *args, **kwargs):

        position = lambda column: 2 * int(Positions[str(column["position"]).upper()] is Positions.LONG) - 1
        quantity = lambda rows: 
        entry = lambda rows:
        dataframe["position"] = dataframe["position"].apply(position)
        columns = [column for column in dataframe.columns if column not in ["position", "quantity", "entry"]]
        dataframe = dataframe.groupby(columns, as_index=False).agg({"quantity": quantity, "entry": entry})





        quantity = lambda cols: (cols["position"].apply(position) * cols["quantity"]).sum()
        entry = lambda cols: cols["entry"]
        function = lambda cols: {"quantity": quantity(cols), "entry": entry(cols)}

        dataframe = dataframe.groupby(columns, as_index=False).apply(function, axis=1, result_type="expand")




        #.rename(columns={None: "holdings"})

        dataframe = dataframe.where(dataframe["holdings"] != 0).dropna(how="all", inplace=False)

        position = lambda cols: str(Positions.LONG.name).lower() if cols["holdings"] > 0 else str(Positions.SHORT.name).lower()
        quantity = lambda cols: np.abs(cols["holdings"])
        function = lambda cols: {"position": position(cols), "quantity": quantity(cols)}
        dataframe = pd.concat([dataframe, dataframe.apply(function, axis=1, result_type="expand")], axis=1)

        dataframe = dataframe.drop("holdings", axis=1, inplace=False)
        return dataframe


class ExposureFiles(object):
    Exposure = ExposureFile

class ExposureHeaders(object):
    Exposure = ExposureHeader



