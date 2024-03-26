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
from collections import namedtuple as ntuple

from support.pipelines import CycleProducer, Consumer
from support.processes import Reader, Writer
from support.tables import DataframeTable
from support.files import DataframeFile

from finance.variables import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetHoldings", "TargetStatus", "TargetTable", "TargetWriter", "TargetReader"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {option: str for option in list(map(str, Securities.Options))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}


class TargetFile(DataframeFile, header=INDEX | VALUES): pass
class TargetTable(DataframeTable):
    def read(self, *args, **kwargs):
        pass

    def write(self, content, *args, **kwargs):
        pass


TargetStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)
class TargetHoldings(tuple):
    Contract = ntuple("Contract", "ticker expire")
    Option = ntuple("Option", "instrument position strike")
    Holding = ntuple("Holding", "contract option")

    def __new__(cls, records):
        assert isinstance(records, list)
        assert all([isinstance(record, dict) for record in records])
        function = lambda record: record["strike"]
        records = sorted(records, key=function, reverse=False)
        contracts = [cls.Contract(record["ticker"], record["expire"]) for record in records]
        options = [cls.Option(record["instrument"], record["position"], record["strike"]) for record in records]
        holdings = [cls.Holding(contract, option) for contract, option in zip(contracts, options)]
        return super().__new__(cls, holdings)

    def __bool__(self): pass
    def __add__(self, other): pass
    def __sub__(self, other): pass

    @property
    def ceiling(self): pass
    @property
    def floor(self): pass

    @property
    def bull(self): pass
    @property
    def bear(self): pass

    @property
    def instruments(self): pass
    @property
    def positions(self): pass
    @property
    def strikes(self): pass


class TargetReader(Reader, CycleProducer, ABC):
    def execute(self, *args, **kwargs):
        melt = dict(name="security", variable="strike", columns=list(map(str, Securities.Options)))
        columns = ["security", "strike", "ticker", "expire", "date"]
        market = self.read(*args, **kwargs)
        if bool(market.empty):
            return
        portfolio = self.melt(market, *args, **melt, **kwargs)
        portfolio = portfolio[columns]
        portfolio["quantity"] = 1

        ### ????? ###

        yield portfolio

    def read(self, *args, **kwargs):
        with self.source.mutex:
            mask = self.source.table["status"] == TargetStatus.PURCHASED
            dataframe = self.source.table.where(mask)
            self.source.remove(dataframe, *args, **kwargs)
            return dataframe


class TargetWriter(Writer, Consumer, ABC):
    def __init__(self, *args, valuation, priority, **kwargs):
        assert callable(priority)
        super().__init__(*args, **kwargs)
        self.__priority = priority
        self.__valuation = valuation

    def market(self, valuations, *args, liquidity=None, apy=None, **kwargs):
        pivot = dict(columns="scenario", values=["apy", "npv", "cost", "size", "tau"])
        function = lambda cols: (np.min(cols.values) * liquidity).apply(np.floor).astype(np.int32)
        liquidity = liquidity if liquidity is not None else 1
        apy = apy if apy is not None else 0
        mask = valuations["valuation"] == str(self.valuation.name).lower()
        valuations = valuations.where(mask).dropna(axis=0, how="all")
        scenarios = set(valuations["scenario"].values)
        market = self.pivot(valuations, *args, **pivot, **kwargs)
        columns = [(scenario, "size") for scenario in scenarios]
        market["liquidity"] = market[columns].apply(function)
        mask = market["liquidity"] > liquidity & market["apy"] > apy
        market = market.where(mask).dropna(axis=0, how="all")
        market = market.reset_index(drop=True, inplace=False)
        return market

    def prioritize(self, market, *args, **kwargs):
        market["priority"] = market.apply(self.priority)
        market = market.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        mask = market["priority"] > 0
        market = market.where(mask).dropna(axis=0, how="all")
        market = market.reset_index(drop=True, inplace=False)
        return market

    def write(self, dataframe, *args, **kwargs):
        with self.source.mutex:
            maximum = np.max(self.source.table.index.values)
            dataframe["status"] = TargetStatus.PROSPECT
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            dataframe = dataframe.set_index(dataframe.index + maximum, drop=True, inplace=False)
            self.source.concat(dataframe, *args, **kwargs)
            self.source.sort("priority", reverse=False)

    @property
    def priority(self): return self.__priority
    @property
    def valuation(self): return self.__valuation



