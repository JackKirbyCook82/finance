# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
from abc import ABC
from enum import IntEnum

import pandas as pd
from support.pipelines import CycleProducer, Consumer
from support.processes import Reader, Writer
from support.tables import DataframeTable
from support.files import DataframeFile

from finance.variables import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetReader", "TargetWriter", "TargetTable", "TargetStatus", "HoldingFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


TargetStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)
TargetIndex = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
TargetColumns = {"quantity": np.int32}

class HoldingFile(DataframeFile, variable="holding", index=TargetIndex, columns=TargetColumns): pass
class TargetTable(DataframeTable): pass


class TargetReader(Reader, CycleProducer, ABC):
    def execute(self, *args, **kwargs):
        market = self.read(*args, **kwargs)
        if bool(market.empty):
            return
        contracts = ["ticker", "expire", "date"]
        securities = [security for security in list(map(str, iter(Securities))) if security in market.columns]
        market = market[contracts + securities].stack()
        market = market.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        market["instrument"] = market["security"].apply(lambda security: str(Securities[security].instrument.name).lower())
        market["position"] = market["security"].apply(lambda security: str(Securities[security].position.name).lower())

        print(market)
        raise Exception()

        yield

    def read(self, *args, **kwargs):
        with self.source.mutex:
            if not bool(self.source):
                return pd.DataFrame()
            mask = self.source.table["status"] == TargetStatus.PURCHASED
            dataframe = self.source.table.where(mask).dropna(how="all", inplace=False)
            self.source.remove(dataframe, *args, **kwargs)
            return dataframe


class TargetWriter(Writer, Consumer, ABC):
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
        dataframe["status"] = TargetStatus.PROSPECT
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



