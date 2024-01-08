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
from itertools import chain, zip_longest
from datetime import datetime as Datetime
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from support.pipelines import Processor, Writer, Reader
from support.tables import DataframeTable

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator", "TargetWriter", "TargetReader", "TargetTerminal", "TargetTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Status = IntEnum("Status", ["PROSPECTED", "PENDING", "ACQUIRED", "ABANDONED"], start=1)


class TargetsQuery(ntuple("Query", "current ticker expire targets")):
    def __str__(self): return "{}|{}, {:.0f}".format(self.ticker, self.expire.strftime("%Y-%m-%d"), len(self.targets.index))


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
        LOGGER.info("Targets: {}[{}]".format(repr(self), str(query)))
        yield query


class TargetWriter(Writer):
    def execute(self, content, *args, **kwargs):
        targets = content.targets if isinstance(content, TargetsQuery) else content
        assert isinstance(targets, pd.DataFrame)
        if bool(targets.empty):
            return
        self.write(targets, *args, **kwargs)
        LOGGER.info("Targets: {}[{}]".format(repr(self), str(self.destination)))
        print(self.destination.table)


class TargetReader(Reader):
    def execute(self, *args, **kwargs):
        pass


class TargetTable(DataframeTable):
    def __str__(self): return "{:,.02f}%, ${:,.0f}|${:,.0f}, {:.0f}|{:.0f}".format(self.apy * 100, self.npv, self.cost, *self.tau)

    def execute(self, dataframe, *args, funds=None, tenure=None, **kwargs):
        dataframe["status"] = Status.PROSPECTED
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


class TargetTerminal(object):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__horizontal = "  /  "
        self.__vertical = "\n"

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        columns = [str(member.name).lower().title() for member in Status]
        columns = ODict([(string, [self.create(*args, **kwargs) for _ in range(count + 5)]) for count, string in enumerate(columns)])
        columns = [self.column(title, *args, frames=frames, **kwargs) for title, frames in columns.items()]
        window = self.window(repr(self), columns=columns)
        while True:
            event, values = window.read()
            if event == gui.WINDOW_CLOSED:
                break
        window.close()

    def create(self, *args, **kwargs):
        header = self.header("strategy", "size")
        securities = [self.security("instrument", "position", "strike"), self.security("instrument", "position", "strike")]
        strategy = self.strategy("ticker", "expire", *securities)
        valuation = self.valuation("apy", "tau", "npv", "cost")
        body = self.body(strategy, valuation)
        footer = self.footer("current", "accept", "reject")
        frame = self.frame(header, body, footer)
        return frame

    @staticmethod
    def header(strategy, size):
        left = gui.Text(strategy, font="Arial 10", size=(25, 1))
        right = gui.Text(size, font="Arial 10", size=(10, 1))
        return [left, gui.VerticalSeparator(), right]

    @staticmethod
    def body(strategy, valuation):
        left = gui.Text(strategy, font="Arial 10", size=(25, 3))
        right = gui.Text(valuation, font="Arial 10", size=(10, 3))
        return [left, gui.VerticalSeparator(), right]

    @staticmethod
    def footer(current, accept, reject):
        current = gui.Text(current, font="Arial 10", size=(25, 3))
        accept = gui.Button(accept, font="Arial 8")
        reject = gui.Button(reject, font="Arial 8")
        return [current, gui.VerticalSeparator(), accept, reject]

    @staticmethod
    def frame(header, body, footer):
        layout = [header, [gui.HorizontalSeparator()], body, [gui.HorizontalSeparator()], footer]
        return gui.Frame("", layout, size=(335, 135))

    @staticmethod
    def column(title, *args, frames=[], **kwargs):
        title = gui.Text(title, font="Arial, 10 bold")
        layout = [[frame] for frame in iter(frames)]
        scrollable = gui.Column(layout, vertical_alignment="top", scrollable=True, vertical_scroll_only=True, size=(350, 865))
        layout = [[title], [gui.HorizontalSeparator()], [scrollable]]
        column = gui.Column(layout, vertical_alignment="top")
        return column

    @staticmethod
    def window(title, *args, columns=[], **kwargs):
        columns = [[column, gui.VerticalSeparator()] for column in columns]
        layout = list(chain(*columns))[:-1]
        window = gui.Window(title, [layout])
        return window

    def security(self, instrument, position, strike): return str(self.horizontal).join([instrument, position, strike])
    def strategy(self, ticker, expire, *securities): return str(self.vertical).join([str(self.horizontal).join([ticker, expire]), *securities])
    def valuation(self, apy, tau, npv, cost): return str(self.vertical).join([str(self.horizontal).join([apy, tau]), npv, cost])

    @property
    def horizontal(self): return self.__horizontal
    @property
    def vertical(self): return self.__vertical
    @property
    def name(self): return self.__name



