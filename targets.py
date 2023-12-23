# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging

import numpy as np
import pandas as pd
from support.pipelines import Processor
from datetime import datetime as Datetime

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetTerminal"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class TargetTerminal(Processor):
    def __init__(self, *args, size=None, tenure=None, **kwargs):
        super().__init__(*args, **kwargs)
        index = ["ticker", "date", "expire", "strategy"] + list(map(str, Securities.Options))
        columns = ["current", "apy", "npv", "cost", "tau", "size"]
        self.__targets = pd.DataFrame(columns=index + columns)
        self.__columns = columns
        self.__index = index
        self.__tenure = tenure
        self.__size = size

    def execute(self, query, *args, **kwargs):
        targets = query.valuations[Valuations.Arbitrage.Minimum]
        if bool(targets.empty):
            return
        targets["current"] = query.current
        targets["apy"] = targets["apy"].round(2)
        targets["npv"] = targets["npv"].round(2)
        targets = self.parser(targets, *args, **kwargs)
        targets = pd.concat([self.targets, targets], axis=0)
        targets = targets.where(targets["current"] - Datetime.now() > self.tenure) if self.tenure is not None else targets
        targets = targets.sort_values(["apy", "npv"], axis=0, ascending=False, inplace=False, ignore_index=False)
        targets = targets.head(self.size) if self.size is not None else targets
        self.targets = targets

        if bool(targets.empty): return
        print(targets)
        raise Exception()

    def parser(self, dataframe, *args, apy=None, funds=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.where(dataframe["cost"] >= funds) if funds is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.drop_duplicates(subset=self.index, keep="last", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe[self.index + self.columns]

    @property
    def targets(self): return self.__targets
    @targets.setter
    def targets(self, targets): self.__targets = targets

    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index
    @property
    def tenure(self): return self.__tenure
    @property
    def size(self): return self.__size




