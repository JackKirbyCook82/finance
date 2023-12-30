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
from support.pipelines import Calculator
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Table

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator", "TargetTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
INDEX = ["ticker", "date", "expire", "strategy"] + list(map(str, Securities.Options))
COLUMNS = ["current", "apy", "npv", "cost", "tau", "size", "liquid"]


class TargetsQuery(ntuple("Query", "current ticker expire targets")):
    def __str__(self): return "{}|{}, {:.0f}".format(self.ticker, self.expire.strftime("%Y-%m-%d"), len(self.targets.index))


class TargetCalculator(Calculator):
    def __init__(self, *args, name, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__valuation = valuation
        self.__columns = COLUMNS
        self.__index = INDEX

    def execute(self, query, *args, liquidity=None, **kwargs):
        if not bool(query.valuations):
            return
        valuations = query.valuations[self.valuation]
        if bool(valuations.empty):
            return
        liquidity = liquidity if liquidity is not None else 1
        valuations["current"] = query.current
        valuations["apy"] = valuations["apy"].round(2)
        valuations["npv"] = valuations["npv"].round(2)
        valuations["tau"] = valuations["tau"].astype(np.int32)
        valuations["size"] = valuations["size"].apply(np.floor).astype(np.int32)
        valuations["liquid"] = (valuations["size"] * liquidity).astype(np.int32)
        targets = self.parser(valuations, *args, **kwargs)
        if bool(targets.empty):
            return
        query = TargetsQuery(query.current, query.ticker, query.expire, targets)
        LOGGER.info("Targets: {}[{}]".format(repr(self), str(query)))
        yield query

    def parser(self, dataframe, *args, apy=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe[self.index + self.columns]

    @property
    def valuation(self): return self.__valuation
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class TargetTable(Table, index=INDEX, columns=COLUMNS):
    def __str__(self): return "${:,.0f}|${:,.0f}".format(self.npv, self.cost)

    def execute(self, content, *args, **kwargs):
        targets = content.targets if isinstance(content, TargetsQuery) else content
        assert isinstance(targets, pd.DataFrame)
        if bool(targets.empty):
            return
        with self.mutex:
            targets = targets[self.index + self.columns]
            targets = pd.concat([self.table, targets], axis=0)
            targets = self.parser(targets, *args, **kwargs)
            self.table = targets
            LOGGER.info("Targets: {}[{}]".format(repr(self), str(self)))
            print(self.table)

    @staticmethod
    def parser(dataframe, *args, limit=None, tenure=None, **kwargs):
        dataframe = dataframe.where(dataframe["current"] - Datetime.now() < tenure) if tenure is not None else dataframe
        dataframe = dataframe.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.head(limit) if limit is not None else dataframe
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def tau(self): return self.table["tau"].min(), self.table["tau"].max()
    @property
    def npv(self): return self.table["npv"] @ self.table["liquid"]
    @property
    def cost(self): return self.table["cost"] @ self.table["liquid"]
    @property
    def size(self): return self.table["liquid"].sum()


#    def __str__(self):
#        apy, cost, suffix = (self.apy * 100, self.cost, "")
#        cost, suffix = (cost / 1000, "K") if cost >= 1000 else (cost, suffix)
#        cost, suffix = (cost / 1000, "M") if cost >= 1000 else (cost, suffix)
#        cost, suffix = (cost / 1000, "B") if cost >= 1000 else (cost, suffix)
#        return "{:,.0f}%|${:,.0f}{}".format(apy * 100, cost, suffix)

#    def terminal(self, *args, **kwargs):
#        title = repr(self)
#        window = gui.Window(title, layout=[])
#        while True:
#            event, values = window.read()
#            if event == gui.WINDOW_CLOSED:
#                break
#        window.close()

#    @staticmethod
#    def frame(row):
#        title = lambda string: "|".join([str(substring).title() for substring in str(string).split("|")])
#        strategy = title(row["strategy"])
#        ticker = str(row["ticker"]).upper()
#        expire = str(row["expire"].strftime("%Y/%m/%d"))
#        options = {title(option): getattr(row, option) for option in list(Securities.Options)}
#        options = ["|".join([str(option), str(strike)]) for option, strike in options.items()]
#        layout = []
#        frame = gui.Frame(title, layout)

#    @property
#    def weights(self):
#        cost = self.table["cost"] / self.table["cost"].sum()
#        size = self.table["size"] / self.table["size"].sum()
#        weights = cost * size
#        weights = weights / weights.sum()
#        return weights

#    @property
#    def apy(self): return self.table["apy"] @ self.weights





