# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import time
import logging
import numpy as np
import pandas as pd
import PySimpleGUI as gui
from enum import IntEnum
from itertools import chain
from datetime import datetime as Datetime
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from support.pipelines import Processor, Writer, Terminal
from support.tables import DataframeTable

from finance.securities import Securities
from finance.strategies import Strategies
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator", "TargetWriter", "TargetTerminal", "TargetTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Status = IntEnum("Status", ["PROSPECTS", "PURSING", "PENDING", "ACQUIRED", "ABANDONED"], start=1)


class Contract(ntuple("Contract", "ticker expire")):
    def __str__(self): return f"{str(self.ticker)} @ {self.expire.strftime('%Y-%m-%d')}"

class Option(ntuple("Option", "instrument position strike")):
    def __str__(self): return f"{str(self.position.name).upper()} {str(self.instrument.name).upper()} @ ${self.strike:.02f}"

class Strategy(ntuple("Strategy", "position spread instrument")):
    def __str__(self): return "{} {} {}".format(*[str(value.name).upper() if bool(value) else "" for value in iter(self)]).strip()

class Valuation(ntuple("Valuation", "apy tau npv cost")):
    def __str__(self): return f"{self.tau:.0f} @ {self.apy * 100:.0f}%\n${self.npv:,.0f}\n${self.cost:,.0f}"

class Target(ntuple("Target", "index status current contract options strategy valuation size")):
    pass

class Action(ntuple("Action", "name status index")):
    def __init_subclass__(cls, status): cls.__status__ = status
    def __new__(cls, index): return super().__new__(cls, cls.__name__, cls.__status__, index)
    def __str__(self): return f"{int(self.index):.0f}|{int(self.status):.0f}"


#    def __new__(cls, index, row):
#        assert isinstance(row, dict)
#        contract = Contract(row["ticker"], row["expire"])
#        options = {option: row.get(str(option), np.NaN) for option in list(Securities.Options)}
#        options = [Option(option.instrument, option.position, strike) for option, strike in options.items() if not np.isnan(strike)]
#        strategy = Strategies[str(row["strategy"])]
#        strategy = Strategy(strategy.position, strategy.spread, strategy.instrument)
#        valuation = Valuation(row["apy"], row["tau"], row["npv"], row["cost"])
#        return super().__new__(cls, index, row["status"], row["current"], contract, options, strategy, valuation, row["size"])

#    @property
#    def table(self):
#        contract = dict(ticker=self.contract.ticker, expire=self.contract.expire)
#        options = {Securities[(option.instrument, option.position)]: option.strike for option in self.options}
#        strategy = Strategies[(self.strategy.spread, self.strategy.instrument, self.strategy.position)]
#        valuations = dict(apy=self.valuation.apy, tau=self.valuation.tau, npv=self.valuation.npv, cost=self.valuation.cost)
#        row = dict(index=self.index, status=self.status, current=self.current, size=self.size, strategy=str(strategy))
#        row = row | contract | options | valuations
#        dataframe = pd.DataFrame([row]).set_index("index", drop=True, inplace=False)
#        return dataframe


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
    def __str__(self): return f"{self.apy * 100:,.02f}%, ${self.npv:,.0f}|${self.cost:,.0f}, {self.tau[0]:.0f}|{self.tau[-1]:.0f}"

    def execute(self, dataframe, *args, funds=None, tenure=None, **kwargs):
        dataframe["status"] = Status.PROSPECTS
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


#    def execute(self, targets, *args, tenure=None, **kwargs):
#        titles = [str(status.name).upper() for status in Status]
#        columns = ODict([(title, []) for title in titles])
#        for status, target in self.generator(targets):
#            title = str(status.name).upper()
#            termtime = (target.current - Datetime.now())
#            lifetime = min(tenure - termtime, 0) if tenure is not None else None
#            size = target.size
#            header = self.header(target.strategy)
#            body = self.body(target.stock, target.options, target.valuation)
#            footer = self.footer("Accept", "Reject")
#            frame = self.frame()
#            columns[title].append(frame)
#        columns = [self.column(title, frames) for title, frames in columns.items()]
#        return self.window(repr(self), columns=columns)

class Pursue(Action, status=Status.PURSING): pass
class Placed(Action, status=Status.PENDING): pass
class Success(Action, status=Status.ACQUIRED): pass
class Failure(Action, status=Status.ABANDONED): pass
class Abandon(Action, status=Status.ABANDONED): pass

class TargetTerminal(Terminal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prospects = {Status.PROSPECTS: [Pursue, Abandon]}
        placed = {Status.PURSING: [Placed, Abandon]}
        pending = {Status.PENDING: [Success, Failure]}
        actions = prospects | placed | pending
        self.__actions = actions

    def process(self, *args, **kwargs):
        time.sleep(5)
        targets = self.stack.read()
        window = self.execute(targets, *args, **kwargs)
        while True:
            event, values = window.read()
            if event == gui.WINDOW_CLOSED:
                break
            print(event)
            print(values)
        window.close()

    def execute(self, *args, tenure=None, **kwargs):
        pass

    @staticmethod
    def frame(strategy, contract, options, valuation, actions=[]):
        assert isinstance(actions, list)
        header = gui.Text(str(strategy), font="Arial 10 bold", size=(22, 1))
        securities = list(map(str, [contract] + list(options)))
        securities = gui.Text("\n".join(securities), font="Arial 10", size=(22, 3))
        valuation = gui.Text(valuation, font="Arial 10")
        body = [securities, gui.VerticalSeparator(), valuation]
        footer = [gui.Button(action.name, key=str(action), font="Arial 8") for action in actions]
        layout = [header, [gui.HorizontalSeparator()], body, [gui.HorizontalSeparator()], footer]
        return gui.Frame("", layout, size=(310, 140))

    @staticmethod
    def column(title, frames=[]):
        title = gui.Text(title, font="Arial, 10 bold")
        layout = [[frame] for frame in iter(frames)]
        scrollable = gui.Column(layout, vertical_alignment="top", scrollable=True, vertical_scroll_only=True, size=(325, 875))
        layout = [[title], [gui.HorizontalSeparator()], [scrollable]]
        return gui.Column(layout, vertical_alignment="top")

    @staticmethod
    def window(title, columns=[]):
        columns = [[column, gui.VerticalSeparator()] for column in columns]
        layout = list(chain(*columns))[:-1]
        return gui.Window(title, [layout])

    @property
    def actions(self): return self.__actions



