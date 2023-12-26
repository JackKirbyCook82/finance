# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
import multiprocessing
from support.pipelines import Calculator
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class TargetsQuery(ntuple("Query", "current ticker expire targets")):
    def __str__(self): return "{}|{}, {:.0f}".format(self.ticker, self.expire.strftime("%Y-%m-%d"), len(self.targets.index))


class TargetCalculator(Calculator):
    def __init__(self, *args, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, **kwargs)
        index = ["ticker", "date", "expire", "strategy"] + list(map(str, Securities.Options))
        columns = ["current", "apy", "npv", "cost", "tau", "size"]
        self.__valuation = valuation
        self.__columns = columns
        self.__index = index

    def execute(self, query, *args, **kwargs):
        valuations = query.valuations[self.valuation]
        if bool(valuations.empty):
            return
        valuations["current"] = query.current
        valuations["apy"] = valuations["apy"].round(2)
        valuations["npv"] = valuations["npv"].round(2)
        targets = self.parser(valuations, *args, **kwargs)
        if bool(targets.empty):
            return

        print(targets)
        raise Exception()

        query = TargetsQuery(query.current, query.ticker, query.expire, targets)
        LOGGER.info("Targets: {}[{}]".format(repr(self), str(query)))
        yield query

    def parser(self, dataframe, *args, apy=None, funds=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.where(dataframe["cost"] >= funds) if funds is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.drop_duplicates(subset=self.index, keep="last", inplace=False)
        dataframe = dataframe.sort_values(["apy", "npv"], axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe[self.index + self.columns]

    @property
    def valuation(self): return self.__valuation
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class TargetStack(object):
    def __init__(self, *args, tenure, size, **kwargs):
        index = ["ticker", "date", "expire", "strategy"] + list(map(str, Securities.Options))
        columns = ["current", "apy", "npv", "cost", "tau", "size"]
        self.__targets = pd.DataFrame(columns=index + columns)
        self.__mutex = multiprocessing.Lock()
        self.__columns = columns
        self.__index = index
        self.__tenure = tenure
        self.__size = size

    def execute(self, content, *args, **kwargs):
        targets = content.targets if isinstance(content, TargetsQuery) else content
        assert isinstance(targets, pd.DataFrame)
        if bool(targets.empty):
            return
        with self.mutex:
            targets = targets[self.index + self.columns]
            targets = pd.concat([self.targets, targets], axis=0)
            targets = self.parser(targets, *args, **kwargs)
            self.targets = targets

    def parser(self, dataframe, *args, **kwargs):
        dataframe = dataframe.where(dataframe["current"] - Datetime.now() > self.tenure) if self.tenure is not None else dataframe
        dataframe = dataframe.sort_values(["apy", "npv"], axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.head(self.size) if self.size is not None else dataframe
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @property
    def targets(self): return self.__targets
    @targets.setter
    def targets(self, targets): self.__targets = targets

    @property
    def length(self): return len(self.targets.index)
    @property
    def empty(self): return bool(self.targets.empty)

    @property
    def mutex(self): return self.__mutex
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index
    @property
    def tenure(self): return self.__tenure
    @property
    def size(self): return self.__size



