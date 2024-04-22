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

from support.pipelines import Producer, Processor, Consumer
from support.processes import Process, Reader, Writer
from support.tables import Tables, Options
from support.queues import Queues
from support.files import Files

from finance.variables import Contract, Securities, Strategies, Scenarios, Instruments, Positions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator", "HoldingReader", "HoldingWriter", "HoldingTable", "HoldingFile", "HoldingQueue", "HoldingStatus"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


HoldingStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)

holding_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))}
holding_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))})
holding_formats.update({("priority", ""): lambda column: f"{column * 100:.02f}"})
holding_formats.update({("status", ""): lambda column: str(HoldingStatus(int(column)).name).lower()})
holding_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=holding_formats, numbers=lambda column: f"{column:.02f}")
holding_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
holding_columns = {"quantity": np.int32}


class HoldingFile(Files.Dataframe, variable="holdings", index=holding_index, columns=holding_columns): pass
class HoldingTable(Tables.Dataframe, options=holding_options): pass
class HoldingQueue(Queues.FIFO, variable="contract"): pass


class HoldingReader(Reader, Producer, ABC):
    def execute(self, *args, **kwargs):
        valuations = self.read(*args, **kwargs)
        if self.empty(valuations):
            return
        contract = self.contract(valuations, *args, **kwargs)
        options = self.options(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        stocks = stocks.dropna(how="all", axis=1)
        securities = pd.concat([contract, options, stocks], axis=1, ignore_index=False)
        holdings = self.holdings(securities, *args, **kwargs)
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            contract = Contract(ticker, expire)
            yield dict(contract=contract, holdings=dataframe)

    @staticmethod
    def contract(dataframe, *args, **kwargs):
        contract = [column for column in list(holding_index.keys()) if column in dataframe.columns]
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
        index = list(holding_index.keys())
        columns = list(holding_columns.keys())
        dataframe = dataframe.groupby(index, as_index=False)[columns].sum()
        return dataframe

    def read(self, *args, **kwargs):
        with self.source.mutex:
            if not bool(self.source):
                return pd.DataFrame()
            mask = self.source.table["status"] == HoldingStatus.PURCHASED
            dataframe = self.source.table.where(mask).dropna(how="all", inplace=False)
            self.source.remove(dataframe, *args, **kwargs)
            return dataframe


class HoldingWriter(Writer, Consumer, ABC):
    def __init__(self, *args, valuation, liquidity, priority, **kwargs):
        assert callable(liquidity) and callable(priority)
        super().__init__(*args, **kwargs)
        self.valuation = valuation
        self.liquidity = liquidity
        self.priority = priority

    def market(self, dataframe, *args, **kwargs):
        dataframe = dataframe.reset_index(drop=True, inplace=False)
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


class HoldingCalculator(Process, Processor):
    def execute(self, query, *args, **kwargs):
        holdings = query["holdings"]
        if self.empty(holdings):
            return
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        holdings = self.holdings(securities, *args, *kwargs)
        yield query | dict(holdings=holdings)

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



