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

from support.pipelines import CycleProducer, Producer, Processor, Consumer
from support.processes import Loader, Saver, Reader, Writer
from support.tables import Tables, Options
from support.files import Files

from finance.variables import Securities, Strategies, Scenarios

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
holding_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=holding_formats, numbers=lambda column: f"{column:.02f}")
holding_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
holding_columns = {"quantity": np.int32}


class HoldingFile(Files.Dataframe, variable="holdings", index=holding_index, columns=holding_columns): pass
class HoldingTable(Tables.Dataframe, options=holding_options): pass
class HoldingLoader(Loader, Producer, title="Loaded"): pass
class HoldingSaver(Saver, Consumer, title="Saved"): pass


class HoldingCalculator(Processor):
    def execute(self, query, *args, **kwargs):
        holdings = query["holdings"]
        assert isinstance(holdings, pd.DataFrame)


class HoldingReader(Reader, CycleProducer, ABC):
    def __init_subclass__(cls, *args, variable, **kwargs): cls.variable = variable

    def execute(self, *args, **kwargs):
        holdings = self.read(*args, **kwargs)
        index = list(holding_index.keys())
        columns = list(holding_columns.keys())
        if bool(holdings.empty):
            return

        print(holdings)
        raise Exception()

        instrument = lambda security: str(Securities[security].instrument.name).lower()
        position = lambda security: str(Securities[security].position.name).lower()
        contracts = [column for column in index if column in holdings.columns]
        securities = [security for security in list(map(str, iter(Securities))) if security in holdings.columns]
        holdings = holdings[contracts + securities].stack()
        holdings = holdings.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        holdings["instrument"] = holdings["security"].apply(instrument)
        holdings["position"] = holdings["security"].apply(position)
        holdings["quantity"] = 1
        holdings = holdings[index + columns]
        holdings = holdings.groupby(index, as_index=False)[columns].sum()
        yield dict(holdings=holdings)

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



