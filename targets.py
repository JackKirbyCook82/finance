# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
from support.pipelines import Calculator
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
    def __init__(self, *args, name, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, name=name, **kwargs)
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



