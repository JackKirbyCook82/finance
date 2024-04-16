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
from itertools import product, chain

from support.pipelines import CycleProducer, Processor, Consumer
from support.processes import Loader, Saver, Reader, Writer
from support.tables import Tables, Options
from support.files import Files

from finance.variables import Contract, Securities, Strategies, Scenarios, Instruments, Positions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator", "HoldingReader", "HoldingWriter", "HoldingLoader", "HoldingSaver", "HoldingTable", "HoldingFile", "HoldingStatus"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


HoldingStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)

holding_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))}
holding_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))})
holding_formats.update({("priority", ""): lambda column: f"{column * 100:.02f}"})
holding_formats.update({("status", ""): lambda column: str(HoldingStatus(int(column)).name).lower()})
query_function = lambda folder: {"contract": Contract.fromstring(folder)}
folder_function = lambda query: query["contract"].tostring()
holding_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=holding_formats, numbers=lambda column: f"{column:.02f}")
holding_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
holding_columns = {"quantity": np.int32}


class HoldingFile(Files.Dataframe, variable="holdings", index=holding_index, columns=holding_columns): pass
class HoldingTable(Tables.Dataframe, options=holding_options): pass
class HoldingSaver(Saver, Consumer, folder=folder_function, title="Saved"): pass
class HoldingLoader(Loader, CycleProducer, query=query_function, title="Loaded"): pass


class HoldingReader(Reader, CycleProducer, ABC):
    def __init_subclass__(cls, *args, variable, **kwargs): cls.variable = variable

    def execute(self, *args, **kwargs):
        valuations = self.read(*args, **kwargs)
        if bool(valuations.empty):
            return
        contract = self.contract(valuations, *args, **kwargs)
        options = self.options(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        stocks = stocks.dropna(how="all", axis=1)
        securities = pd.concat([contract, options, stocks], axis=1, ignore_index=False)
        holdings = self.holdings(securities, *args, **kwargs)
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            contract = Contract(ticker, expire)
            yield dict(contract=contract, holdings=holdings)

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

    @property
    def source(self): return super().source[self.variable]
    def read(self, *args, **kwargs):
        with self.source.mutex:
            if not bool(self.source):
                return pd.DataFrame()
            mask = self.source.table["status"] == HoldingStatus.PURCHASED
            dataframe = self.source.table.where(mask).dropna(how="all", inplace=False)
            self.source.remove(dataframe, *args, **kwargs)
            return dataframe


class HoldingWriter(Writer, Consumer, ABC):
    def __init_subclass__(cls, *args, variable, **kwargs): cls.variable = variable
    def __init__(self, *args, valuation, liquidity, priority, **kwargs):
        assert callable(liquidity) and callable(priority)
        super().__init__(*args, **kwargs)
        self.valuation = valuation
        self.liquidity = liquidity
        self.priority = priority

    def market(self, dataframe, *args, **kwargs):
        dataframe = dataframe.reset_index(drop=False, inplace=False)
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

    @property
    def destination(self): return super().destination[self.variable]
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


class HoldingCalculator(Processor):
    def execute(self, query, *args, **kwargs):
        holdings = query["holdings"]
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        return securities

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        stocks = str(Instruments.STOCK.name).lower()
        stocks = dataframe.index.get_level_values("instrument") == stocks
        stocks = dataframe.iloc[stocks]
        return stocks

    @staticmethod
    def options(dataframe, *args, **kwargs):
        puts = str(Instruments.PUT.name).lower()
        calls = str(Instruments.CALL.name).lower()
        puts = dataframe.index.get_level_values("instrument") == puts
        calls = dataframe.index.get_level_values("instrument") == calls
        options = dataframe.iloc[puts | calls]
        return options

    @staticmethod
    def virtuals(dataframe, *args, **kwargs):
        invert = lambda x: Positions.SHORT if x == Positions.LONG else Positions.LONG
        strike = lambda x: np.round(x, 2).astype(np.float32)
        parameters = lambda record: {"strike": strike(record["paid"])}
        call = lambda record: {"instrument": Instruments.CALL, "position": record["position"]}
        put = lambda record: {"instrument": Instruments.PUT, "position": invert(record["position"])}
        left = lambda record: record | put(record) | parameters(record)
        right = lambda record: record | call(record) | parameters(record)

        print(dataframe)
        raise Exception()

        virtuals = [[left(record), right(record)] for record in stocks.to_dict("records")]
        virtuals = pd.DataFrame.from_records(list(chain(*virtuals)))
        virtuals["strike"] = virtuals["strike"].apply(strike)
        mask = dataframe["instrument"] != Instruments.STOCK




