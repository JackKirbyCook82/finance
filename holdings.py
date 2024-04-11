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

from support.pipelines import CycleProducer, Consumer
from support.processes import Reader, Writer, Saver
from support.files import Files

from finance.variables import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingReader", "HoldingWriter", "HoldingSaver", "HoldingFile", "HoldingStatus"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


HoldingStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)
HoldingQuantities = {security: 100 for security in list(Securities.Stocks)} | {security: 1 for security in list(Securities.Options)}
HoldingIndex = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
HoldingColumns = {"quantity": np.int32}


class HoldingFile(Files.Dataframe, variable="holdings", index=HoldingIndex, columns=HoldingColumns): pass
class HoldingSaver(Saver, Consumer, files=["holdings"], title="Saved"):
    def execute(self, query, *args, **kwargs):
        for file in self.files:
            self.write(query[file], *args, file=file, **kwargs)


class HoldingReader(Reader, CycleProducer, ABC):
    def execute(self, *args, **kwargs):
        holdings = self.read(*args, **kwargs)
        if bool(holdings.empty):
            return
        instrument = lambda security: str(Securities[security].instrument.name).lower()
        position = lambda security: str(Securities[security].position.name).lower()
        quantity = lambda security: HoldingQuantities[Securities[security]]
        contracts = ["ticker", "expire", "date"]
        securities = [security for security in list(map(str, iter(Securities))) if security in holdings.columns]
        holdings = holdings[contracts + securities].stack()
        holdings = holdings.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        holdings["instrument"] = holdings["security"].apply(instrument)
        holdings["position"] = holdings["security"].apply(position)
        holdings["quantity"] = holdings["security"].apply(quantity)
        yield dict(holdings=holdings)

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
        self.__valuation = valuation
        self.__liquidity = liquidity
        self.__priority = priority

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

    @property
    def valuation(self): return self.__valuation
    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority



