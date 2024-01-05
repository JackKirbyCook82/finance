# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Processor, Writer, Reader
from support.tables import DataframeTable

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator", "TargetWriter", "TargetTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class TargetsQuery(ntuple("Query", "current ticker expire targets")):
    def __str__(self): return "{}|{}, {:.0f}".format(self.ticker, self.expire.strftime("%Y-%m-%d"), len(self.targets.index))


class TargetCalculator(Processor):
    def __init__(self, *args, name=None, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.valuation = valuation

    def execute(self, query, *args, **kwargs):
        if not bool(query.valuations):
            return
        targets = query.valuations[self.valuation]
        if bool(targets.empty):
            return
        targets["current"] = query.current
        targets = self.format(targets, *args, **kwargs)
        targets = self.parser(targets, *args, **kwargs)
        if bool(targets.empty):
            return
        query = TargetsQuery(query.current, query.ticker, query.expire, targets)
        LOGGER.info("Targets: {}[{}]".format(repr(self), str(query)))
        yield query

    @staticmethod
    def format(dataframe, *args, **kwargs):
        dataframe["apy"] = dataframe["apy"].round(2)
        dataframe["npv"] = dataframe["npv"].round(2)
        dataframe["tau"] = dataframe["tau"].astype(np.int32)
        dataframe["size"] = dataframe["size"].apply(np.floor).astype(np.int32)
        return dataframe

    @staticmethod
    def parser(dataframe, *args, liquidity=None, apy=None, funds=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        liquidity = liquidity if liquidity is not None else 1
        liquid = (dataframe["size"] * liquidity).astype(np.int32)
        dataframe = dataframe.loc[dataframe.index.repeat(liquid)]
        dataframe["size"] = 1
        if funds is not None:
            affordable = dataframe["cost"].cumsum() <= funds
            dataframe = dataframe.where(affordable)
            dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class TargetWriter(Writer):
    def execute(self, content, *args, **kwargs):
        targets = content.targets if isinstance(content, TargetsQuery) else content
        assert isinstance(targets, pd.DataFrame)
        if bool(targets.empty):
            return
        self.write(targets, *args, **kwargs)
        LOGGER.info("Targets: {}[{}]".format(repr(self), str(self.destination)))
        print(self.destination.table)
        print(self.destination.targets)


class TargetReader(Reader):
    def execute(self, *args, **kwargs):
        while True:
            pass


class TargetTable(DataframeTable):
    def __str__(self): return "{:,.02f}%, ${:,.0f}|${:,.0f}, {:.0f}|{:.0f}".format(self.apy * 100, self.npv, self.cost, *self.tau)

    @staticmethod
    def parser(dataframe, *args, funds=None, limit=None, tenure=None, **kwargs):
        dataframe = dataframe.where(dataframe["current"] - Datetime.now() < tenure) if tenure is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        if funds is not None:
            affordable = dataframe["cost"].cumsum() <= funds
            dataframe = dataframe.where(affordable)
            dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.head(limit) if limit is not None else dataframe
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @staticmethod
    def format(dataframe, *args, **kwargs):
        dataframe["apy"] = dataframe["apy"].round(2)
        dataframe["npv"] = dataframe["npv"].round(2)
        dataframe["tau"] = dataframe["tau"].astype(np.int32)
        dataframe["size"] = dataframe["size"].apply(np.floor).astype(np.int32)
        return dataframe

    @property
    def header(self): return ["strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @property
    def apy(self): return self.table["apy"] @ (self.table["cost"] / self.table["cost"].sum())
    @property
    def tau(self): return self.table["tau"].min(), self.table["tau"].max()
    @property
    def npv(self): return self.table["npv"].sum()
    @property
    def cost(self): return self.table["cost"].sum()
    @property
    def size(self): return len(self.table.index)

    @property
    def targets(self):
        targets = self.table[~self.table.duplicated(keep="last")]
        size = (targets.reset_index(inplace=False, drop=False)["index"] + 1)
        size = size.diff().fillna(size.values[0]).astype(int)
        targets.reset_index(inplace=True, drop=True)
        targets["size"] = size
        return targets




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





