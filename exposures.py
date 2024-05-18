# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Instruments, Positions
from support.pipelines import Processor
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingFile", "ExposureFile", "OptionFile", "ExposureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


holdings_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
holdings_columns = {"quantity": np.int32}
exposures_columns = {"quantity": np.int32}


class HoldingFile(Files.Dataframe, variable="holdings", index=holdings_index, columns=holdings_columns): pass
class ExposureFile(Files.Dataframe, variable="exposures", index=holdings_index, columns=exposures_columns): pass
class OptionFile(Files.Dataframe, variable="options", index=holdings_index, columns=options_columns): pass


class ExposureCalculator(Processor):
    index = list(holdings_index.keys())
    columns = list(exposures_columns.keys())

    def execute(self, contents, *args, **kwargs):
        holdings = contents["holdings"]
        assert isinstance(holdings, pd.DataFrame)
        if bool(holdings.empty):
            return
        holdings = holdings.reset_index(drop=False, inplace=False)
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposures = self.holdings(securities, *args, *kwargs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        exposures = self.header(exposures)
        yield contents | dict(exposures=exposures)

    def header(self, dataframe):
        index = [column for column in list(self.index) if column in dataframe]
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        return dataframe[self.columns]

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
        return virtuals

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        factor = lambda cols: 2 * int(Positions[str(cols["position"]).upper()] is Positions.LONG) - 1
        position = lambda cols: str(Positions.LONG.name).lower() if cols["holdings"] > 0 else str(Positions.SHORT.name).lower()
        quantity = lambda cols: np.abs(cols["holdings"])
        holdings = lambda cols: (cols.apply(factor, axis=1) * cols["quantity"]).sum()
        function = lambda cols: {"position": position(cols), "quantity": quantity(cols)}
        columns = [column for column in dataframe.columns if column not in ["position", "quantity"]]
        dataframe = dataframe.groupby(columns, as_index=False).apply(holdings).rename(columns={None: "holdings"})
        dataframe = dataframe.where(dataframe["holdings"] != 0).dropna(how="all", inplace=False)
        dataframe = pd.concat([dataframe, dataframe.apply(function, axis=1, result_type="expand")], axis=1)
        dataframe = dataframe.drop("holdings", axis=1, inplace=False)
        return dataframe





