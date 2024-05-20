# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC
from enum import IntEnum
from itertools import product

from finance.variables import Contract, Securities, Strategies, Scenarios
from support.pipelines import Producer, Consumer
from support.tables import Tables, Options
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingTable", "HoldingFile", "HoldingStatus", "HoldingReader", "HoldingWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


HoldingStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)

holdings_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))}
holdings_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))})
holdings_formats.update({("priority", ""): lambda column: f"{column:.02f}"})
holdings_formats.update({("status", ""): lambda column: str(HoldingStatus(int(column)).name).lower()})
holdings_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=holdings_formats, numbers=lambda column: f"{column:.02f}")
holdings_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
holdings_columns = {"quantity": np.int32}


class HoldingFile(Files.Dataframe, variable=("holdings", "holdings"), index=holdings_index, columns=holdings_columns): pass
class HoldingTable(Tables.Dataframe, options=holdings_options):
    def write(self, locator, content, *args, **kwargs):
        locator = self.locate(locator)
        super().write(locator, content, *args, **kwargs)

    def read(self, locator, *args, **kwargs):
        locator = self.locate(locator)
        return super().read(locator, *args, **kwargs)

    def locate(self, locator):
        index, column = locator
        assert isinstance(index, (int, slice))
        if isinstance(index, slice):
            assert index.step is None
            length = len(list(range(index.start, index.stop)))
            index = slice(length) if length > len(self.table.index) else index
        if isinstance(column, str):
            column = (column, "")
            column = list(self.table.columns).index(column)
        elif isinstance(column, tuple):
            assert len(column) == 2
            column = list(self.table.columns).index(column)
        return index, column


class HoldingWriter(Consumer, ABC):
    def __init__(self, *args, destination, valuation, liquidity, priority, capacity=None, **kwargs):
        assert callable(liquidity) and callable(priority)
        super().__init__(*args, **kwargs)
        self.__destination = destination
        self.__valuation = valuation
        self.__liquidity = liquidity
        self.__priority = priority
        self.__capacity = capacity

    def market(self, dataframe, *args, **kwargs):
        index = set(dataframe.columns) - {"scenario", "apy", "npv", "cost"}
        dataframe = dataframe.pivot(index=index, columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        dataframe["liquidity"] = dataframe.apply(self.liquidity, axis=1)
        return dataframe

    def prioritize(self, dataframe, *args, **kwargs):
        dataframe["priority"] = dataframe.apply(self.priority, axis=1)
        dataframe = dataframe.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.where(dataframe["priority"] > 0).dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    def write(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        dataframe["status"] = HoldingStatus.PROSPECT
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        with self.destination.mutex:
            if not bool(self.destination):
                self.destination.table = dataframe
            else:
                index = np.max(self.destination.table.index.values) + 1
                dataframe = dataframe.set_index(dataframe.index + index, drop=True, inplace=False)
                self.destination.concat(dataframe, *args, **kwargs)
            self.destination.sort("priority", reverse=True)
            if bool(self.capacity):
                self.destination.truncate(self.capacity)

    @property
    def destination(self): return self.__destination
    @property
    def valuation(self): return self.__valuation
    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority
    @property
    def capacity(self): return self.__capacity


class HoldingReader(Producer, ABC):
    def __init__(self, *args, source, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__index = list(holdings_index.keys())
        self.__columns = list(holdings_columns.keys())
        self.__valuation = valuation
        self.__source = source

    def execute(self, *args, **kwargs):
        valuations = self.read(*args, **kwargs)
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = valuations.reset_index(drop=False, inplace=False)
        contract = self.contract(valuations, *args, **kwargs)
        options = self.options(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        stocks = stocks.dropna(how="all", axis=1)
        securities = pd.concat([contract, options, stocks], axis=1, ignore_index=False)
        holdings = self.holdings(securities, *args, **kwargs)
        holdings = holdings.groupby(self.index, as_index=False)[self.columns].sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            contract = Contract(ticker, expire)
            yield dict(contract=contract, holdings=dataframe)

    @staticmethod
    def contract(dataframe, *args, **kwargs):
        contract = [column for column in list(holdings_index.keys()) if column in dataframe.columns]
        dataframe = dataframe.droplevel("scenario", axis=1)
        return dataframe[contract]

    @staticmethod
    def options(dataframe, *args, **kwargs):
        securities = list(map(str, iter(Securities)))
        securities = [column for column in securities if column in dataframe.columns]
        dataframe = dataframe.droplevel("scenario", axis=1)
        return dataframe[securities]

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        securities = list(map(str, iter(Securities.Stocks)))
        strategies = lambda cols: list(map(str, Strategies[cols["strategy"]].stocks))
        underlying = lambda cols: np.round(cols["underlying"], decimals=2)
        function = lambda cols: [underlying(cols) if column in strategies(cols) else np.NaN for column in securities]
        dataframe = dataframe.droplevel("scenario", axis=1)
        dataframe = dataframe.apply(function, axis=1, result_type="expand")
        dataframe.columns = securities
        return dataframe

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        instrument = lambda security: str(Securities[security].instrument.name).lower()
        position = lambda security: str(Securities[security].position.name).lower()
        securities = [security for security in list(map(str, iter(Securities))) if security in dataframe.columns]
        contracts = [column for column in dataframe.columns if column not in securities]
        dataframe = dataframe.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        dataframe["instrument"] = dataframe["security"].apply(instrument)
        dataframe["position"] = dataframe["security"].apply(position)
        dataframe["quantity"] = 1
        return dataframe

    def read(self, *args, **kwargs):
        with self.source.mutex:
            if not bool(self.source):
                return pd.DataFrame()
            mask = self.source.table["status"] == HoldingStatus.PURCHASED
            dataframe = self.source.table.where(mask).dropna(how="all", inplace=False)
            self.source.remove(dataframe, *args, **kwargs)
            return dataframe

    @property
    def valuation(self): return self.__valuation
    @property
    def source(self): return self.__source
    @property
    def index(self): return self.__index
    @property
    def columns(self): return self.__columns




