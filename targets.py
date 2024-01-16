# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import PySimpleGUI as gui
from enum import IntEnum
from functools import total_ordering
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.tables import DataframeTable
from support.pipelines import Processor, Writer
from support.windows import Window, Text, Table, Format, Column, Justify

from finance.securities import Securities
from finance.strategies import Strategies
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator", "TargetWriter", "TargetTable", "TargetTerminal"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Status = IntEnum("Status", ["PROSPECT", "PENDING", "PURCHASED"], start=1)


class Strategy(ntuple("Strategy", "spread instrument position")):
    def __str__(self):
        position = str(self.position.name).upper() if bool(self.position) else ""
        instrument = str(self.instrument.name).upper() if bool(self.instrument) else ""
        return " ".join([position, str(self.spread.name).upper(), instrument]).strip()

class Product(ntuple("Product", "ticker expire")):
    def __str__(self): return f"{str(self.ticker).upper()}, {self.expire.strftime('%Y-%m-%d')}"

class Option(ntuple("Option", "instrument position strike")):
    def __str__(self): return f"{str(self.position.name).upper()} {str(self.instrument.name).upper()} @ ${self.strike:.02f}"

@total_ordering
class Valuation(ntuple("Valuation", "profit tau value cost")):
    def __str__(self): return f"{self.tau:.0f} @ {self.profit * 100:.0f}%, ${self.value:,.0f}, ${self.cost:,.0f}"
    def __eq__(self, other): return self.profit == other.profit
    def __lt__(self, other): return self.profit < other.profit

@total_ordering
class Target(ntuple("Target", "identity current strategy product options valuation size")):
    def __eq__(self, other): return self.valuation == other.valuation
    def __lt__(self, other): return self.valuation < other.valuation
    def __hash__(self): return self.identity

    def __init__(self, *args, status, **kwargs): self.__status = status
    def __new__(cls, identity, current, product, strategy, options, valuation, size, *args, **kwargs):
        instance = super().__new__(cls, identity, current, product, strategy, options, valuation, size)
        return instance

    @property
    def status(self): return self.__status
    @status.setter
    def status(self, status): self.__status = status


class TargetsQuery(ntuple("Query", "current ticker expire targets")):
    def __str__(self): return f"{self.ticker}|{self.expire.strftime('%Y-%m-%d')}, {len(self.targets.index):.0f}"


class TargetCalculator(Processor):
    def __init__(self, *args, name=None, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.valuation = valuation

    def execute(self, query, *args, liquidity=None, apy=None, **kwargs):
        if not bool(query.valuations) or bool(query.valuations[self.valuation].empty):
            return
        targets = query.valuations[self.valuation]
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
        targets["current"] = query.current
        query = TargetsQuery(query.current, query.ticker, query.expire, targets)
        LOGGER.info(f"Targets: {repr(self)}[{str(query)}]")
        yield query


class TargetWriter(Writer):
    def execute(self, content, *args, **kwargs):
        targets = content.targets if isinstance(content, TargetsQuery) else content
        assert isinstance(targets, pd.DataFrame)
        if bool(targets.empty):
            return
        self.destination.write(targets, *args, **kwargs)
        LOGGER.info(f"Targets: {repr(self)}[{str(self.destination)}]")
        print(self.destination.table)


class TargetTable(DataframeTable):
    def __str__(self): return f"{self.tau[0]:.0f}|{self.tau[-1]:.0f} @ {self.apy * 100:,.02f}%, ${self.npv:,.0f}|${self.cost:,.0f}, {self.size:.0f}"
    def __iter__(self): return (self.parser(index, record) for index, record in super().__iter__())

    def write(self, dataframe, *args, funds=None, tenure=None, **kwargs):
        super().write(dataframe, *args, **kwargs)
        self.table = self.table.where(self.table["current"] - Datetime.now() < tenure) if tenure is not None else self.table
        self.table = self.table.dropna(axis=0, how="all")
        self.table = self.table.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        if funds is not None:
            columns = [column for column in self.table.columns if column != "size"]
            expanded = self.table.loc[self.table.index.repeat(self.table["size"])][columns]
            expanded = expanded.where(expanded["cost"].cumsum() <= funds)
            expanded = expanded.dropna(axis=0, how="all")
            self.table["size"] = expanded.index.value_counts()
            self.table = self.table.where(self.table["size"].notna())
            self.table = self.table.dropna(axis=0, how="all")
        self.table["size"] = self.table["size"].apply(np.floor).astype(np.int32)
        self.table["tau"] = self.table["tau"].astype(np.int32)
        self.table["apy"] = self.table["apy"].round(2)
        self.table["npv"] = self.table["npv"].round(2)

    @staticmethod
    def parser(identity, record):
        assert isinstance(record, dict)
        strategy = Strategies[str(record["strategy"])]
        strategy = Strategy(strategy.spread, strategy.instrument, strategy.position)
        product = Product(record["ticker"], record["expire"])
        valuation = Valuation(record["apy"], record["tau"], record["npv"], record["cost"])
        options = {option: record.get(str(option), np.NaN) for option in list(Securities.Options)}
        options = [Option(option.instrument, option.position, strike) for option, strike in options.items() if not np.isnan(strike)]
        target = Target(identity, record["current"], strategy, product, options, valuation, record["size"], status=Status.PROSPECT)
        return target

    @property
    def header(self): return ["current", "strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
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


class ContentsTable(Table, justify=Justify.LEFT, height=40, events=True):
    strategy = Column("strategy", 15, lambda target: str(target.strategy))
    ticker = Column("ticker", 10, lambda target: str(target.product.ticker).upper())
    expire = Column("expire", 10, lambda target: target.product.expire.strftime("%Y-%m-%d"))
    options = Column("options", 20, lambda target: "\n".join(list(map(str, target.options))))
    profit = Column("profit", 20, lambda target: f"{target.valuation.profit * 100:.0f}% / YR @ {target.valuation.tau:.0f} DAYS")
    value = Column("value", 10, lambda target: f"${target.valuation.value:,.0f}")
    cost = Column("cost", 10, lambda target: f"${target.valuation.cost:,.0f}")
    size = Column("size", 10, lambda target: f"{target.size:,.0f} CNT")


class LeftText(Text):
    strategy = Format("strategy", 10, lambda target: str(target.strategy))
    product = Format("product", 10, lambda target: str(target.product))
    options = Format("options", 10, lambda target: "\n".join(list(map(str, target.options))))

    @staticmethod
    def layout(*args, content, formats={}, **kwargs):
        return [["\n".join([parser(content) for name, parser in formats.items() if name in ("strategy", "product", "options")])]]


class RightText(Text):
    valuation = Format("valuation", 10, lambda target: str(target.valuation).replace(" ,", "\n"))
    size = Format("size", 10, lambda target: f"{target.size:,.0f} CNT")

    @staticmethod
    def layout(*args, content, formats={}, **kwargs):
        return [["\n".join([parser(content) for name, parser in formats.items() if name in ("valuation", "size")])]]


class TargetsWindow(Window):
    def __init__(self, *args, name, **kwargs):
        prospect = ContentsTable(*args, name="prospect", rows=[], **kwargs)
        pending = ContentsTable(*args, name="pending", rows=[], **kwargs)
        purchased = ContentsTable(*args, name="purchased", rows=[], **kwargs)
        tables = [prospect, pending, purchased]
        super().__init__(*args, name=name, tables=tables, **kwargs)
        self.__prospect = prospect
        self.__pending = pending
        self.__purchased = purchased

    @staticmethod
    def layout(*args, tables, **kwargs):
        assert isinstance(tables, list)
        tabs = [gui.Tab(repr(table), [[table.element]], key=f"--{repr(table)}|tab--") for table in tables]
        return [[gui.TabGroup([tabs], key="--tab|group--")]]

    @property
    def prospect(self): return self.__prospect
    @property
    def pending(self): return self.__pending
    @property
    def purchased(self): return self.__purchased


class TargetWindow(Window):
    def __init__(self, *args, name, target, **kwargs):
        assert isinstance(target, Target)
        Texts = ntuple("Texts", "left right")
        left = LeftText(*args, name="left", content=target, **kwargs)
        right = RightText(*args, name="right", content=target, **kwargs)
        texts = Texts(left, right)
        super().__init__(*args, name=name, texts=texts, **kwargs)

    @staticmethod
    def layout(*args, texts, **kwargs):
        return [[texts.left.element, gui.VerticalSeparator(), texts.right.element]]


class TargetTerminal(object):
    def __repr__(self): return self.name
    def __init__(self, *args, feed, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__targets = list()
        self.__feed = feed

    def __call__(self, *args, **kwargs):
        targets_window = TargetsWindow(*args, name="targets", **kwargs)
        targets = list(iter(self.feed))
        targets = set(self.targets + targets)
        targets = sorted(list(targets), reverse=True)
        targets_window.prospect(*args, rows=targets, **kwargs)
        self.targets = targets
        while True:
            window, event, handles = gui.read_all_windows()
            print(event, handles)
            if event == gui.WINDOW_CLOSED:
                break
            if event is None:
                continue
            for index in handles[event]:
                target = self.targets[index]
                target_window = TargetWindow(*args, name="target", target=target, **kwargs)

    @property
    def targets(self): return self.__targets
    @targets.setter
    def targets(self, targets): self.__targets = targets
    @property
    def feed(self): return self.__feed
    @property
    def name(self): return self.__name








