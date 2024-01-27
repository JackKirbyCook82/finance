# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from enum import IntEnum
from functools import total_ordering
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Processor, Consumer
from support.tables import DataframeTable
from support.files import DataframeFile

from finance.securities import Securities
from finance.strategies import Strategies
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetStatus", "TargetsCalculator", "TargetsWriter", "TargetsFile", "TargetsTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
TargetStatus = IntEnum("Status", ["PROSPECT", "PENDING", "PURCHASED", "ABANDONED"], start=1)


@total_ordering
class Profitability(ntuple("Return", "apy tau")):
    def __str__(self): return f"{self.apy * 100:.0f}% / YR, {self.tau:.0f} DAYS"
    def __eq__(self, other): return self.apy == other.apy
    def __lt__(self, other): return self.apy < other.apy

class Valuation(ntuple("Valuation", "profit cost")):
    def __str__(self): return f"${self.profit:,.0f}, ${self.cost:,.0f}"

class Strategy(ntuple("Strategy", "spread instrument position")):
    def __str__(self):
        position = str(self.position.name).upper() if bool(self.position) else ""
        instrument = str(self.instrument.name).upper() if bool(self.instrument) else ""
        return " ".join([position, str(self.spread.name).upper(), instrument]).strip()

class Product(ntuple("Product", "ticker expire")):
    def __str__(self): return f"{str(self.ticker).upper()} @ {self.expire.strftime('%Y-%m-%d')}"

class Option(ntuple("Option", "instrument position strike")):
    def __str__(self): return f"{str(self.position.name).upper()} {str(self.instrument.name).upper()} @ ${self.strike:.02f}"

class Target(ntuple("Target", "index status current strategy product options profitability valuation size")):
    def __eq__(self, other): return bool(self.index == other.index)
    def __hash__(self): return hash(self.index)


class TargetsQuery(ntuple("Query", "current ticker expire targets")):
    def __str__(self): return f"{self.ticker}|{self.expire.strftime('%Y-%m-%d')}, {len(self.targets.index):.0f}"


class TargetsCalculator(Processor):
    def execute(self, query, *args, liquidity=None, apy=None, **kwargs):
        if not bool(query.arbitrages) or all([dataframe.empty for dataframe in query.arbitrages.values()]):
            return
        targets = query.arbitrages[Valuations.Arbitrage.Minimum]
        liquidity = liquidity if liquidity is not None else 1
        targets["size"] = (targets["size"] * liquidity).apply(np.floor).astype(np.int32)
        targets = targets.where(targets["apy"] >= apy) if apy is not None else targets
        targets = targets.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        targets = targets.dropna(axis=0, how="all")
        if bool(targets.empty):
            return
        assert targets["apy"].min() > 0 and targets["size"].min() > 0
        targets = targets.reset_index(drop=True, inplace=False)
        targets["size"] = targets["size"].astype(np.int32)
        targets["tau"] = targets["tau"].astype(np.int32)
        targets["apy"] = targets["apy"].round(2)
        targets["npv"] = targets["npv"].round(2)
        targets["status"] = TargetStatus.PROSPECT
        targets["current"] = query.current
        query = TargetsQuery(query.current, query.ticker, query.expire, targets)
        LOGGER.info(f"Targets: {repr(self)}[{str(query)}]")
        yield query


class TargetsFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["current", "strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["current", "date", "expire"]


class TargetsTable(DataframeTable):
    def __iter__(self): return (self.parser(index, record) for (index, record) in self.read(list))
    def __str__(self): return f"{self.tau[0]:.0f}|{self.tau[-1]:.0f} @ {self.apy * 100:,.02f}%, ${self.npv:,.0f}|${self.cost:,.0f}, {self.size:.0f}"
    def __setitem__(self, key, value): self.table.at[key] = value
    def __getitem__(self, key): return self.table.loc[key]

    def execute(self, dataframe, *args, funds=None, tenure=None, **kwargs):
        dataframe = super().execute(dataframe, *args, **kwargs)
        dataframe = dataframe.where(dataframe["current"] - Datetime.now() < tenure) if tenure is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        if funds is not None:
            columns = [column for column in dataframe.columns if column != "size"]
            expanded = dataframe.loc[dataframe.index.repeat(dataframe["size"])][columns]
            expanded = expanded.where(expanded["cost"].cumsum() <= funds)
            expanded = expanded.dropna(axis=0, how="all")
            dataframe["size"] = expanded.index.value_counts()
            dataframe = dataframe.where(dataframe["size"].notna())
            dataframe = dataframe.dropna(axis=0, how="all")
        dataframe["size"] = dataframe["size"].apply(np.floor).astype(np.int32)
        dataframe["tau"] = dataframe["tau"].astype(np.int32)
        dataframe["apy"] = dataframe["apy"].round(2)
        dataframe["npv"] = dataframe["npv"].round(2)
        return dataframe

    @staticmethod
    def parser(index, record):
        assert isinstance(record, dict)
        strategy = Strategies[str(record["strategy"])]
        strategy = Strategy(strategy.spread, strategy.instrument, strategy.position)
        product = Product(record["ticker"], record["expire"])
        profitability = Profitability(record["apy"], record["tau"])
        valuation = Valuation(record["npv"], record["cost"])
        options = {option: record.get(str(option), np.NaN) for option in list(Securities.Options)}
        options = [Option(option.instrument, option.position, strike) for option, strike in options.items() if not np.isnan(strike)]
        target = Target(int(index), record["status"], record["current"], strategy, product, options, profitability, valuation, record["size"])
        return target

    @property
    def header(self): return ["status", "current", "strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @property
    def weights(self): return (self.table["cost"] * self.table["size"]) / (self.table["cost"] @ self.table["size"])
    @property
    def tau(self): return self.table["tau"].min(), self.table["tau"].max()
    @property
    def npv(self): return self.table["npv"] @ self.table["size"]
    @property
    def cost(self): return self.table["cost"] @ self.table["size"]
    @property
    def apy(self): return self.table["apy"] @ self.weights
    @property
    def size(self): return self.table["size"].sum()


class TargetsWriter(Consumer):
    def __init__(self, *args, table, file, **kwargs):
        assert isinstance(table, TargetsTable) and isinstance(file, TargetsFile)
        super().__init__(*args, **kwargs)
        self.table = table
        self.file = file

    def execute(self, content, *args, **kwargs):
        targets = content.targets if isinstance(content, TargetsQuery) else content
        assert isinstance(targets, pd.DataFrame)
        if bool(targets.empty):
            return
        targets_file = self.file.path("targets.csv")
        self.file.write(targets, file=targets_file, mode="a")
        LOGGER.info("Saved: {}[{}]".format(repr(self), str(targets_file)))
        self.table.write(targets, *args, **kwargs)
        LOGGER.info(f"Targets: {repr(self)}[{str(self.table)}]")







