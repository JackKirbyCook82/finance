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

from support.pipelines import Consumer
from support.processes import Writer
from support.tables import DataframeTable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetStatus", "TargetTable", "TargetReader", "TargetWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


TargetStatus = IntEnum("Status", ["PROSPECT", "PURCHASED"], start=1)

class TargetTable(DataframeTable): pass
class TargetReader():
    pass


class TargetWriter(Writer, Consumer, ABC):
    def __init__(self, *args, valuation, liquidity, priority, **kwargs):
        assert callable(liquidity) and callable(priority)
        super().__init__(*args, **kwargs)
        self.__valuation = valuation
        self.__liquidity = liquidity
        self.__priority = priority

    def market(self, dataframe, *args, **kwargs):
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        index = set(dataframe.columns) - {"scenario", "apy", "npv", "cost"}
        dataframe = dataframe.pivot(index=index, columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        dataframe["liquidity"] = dataframe.apply(self.liquidity, axis=1)
        return dataframe

    def prioritize(self, dataframe, *args, **kwargs):
        dataframe["priority"] = dataframe.apply(self.priority, axis=1)
        dataframe = dataframe.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.where(dataframe["priority"] > 0).dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    def write(self, dataframe, *args, **kwargs):
        dataframe["status"] = TargetStatus.PROSPECT
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        with self.destination.mutex:
            maximum = np.max(self.destination.table.index.values) if not self.destination.empty else 0
            dataframe = dataframe.set_index(dataframe.index + maximum + 1, drop=True, inplace=False)
            self.destination.concat(dataframe, *args, **kwargs)
            self.destination.sort("priority", reverse=True)

    @property
    def valuation(self): return self.__valuation
    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority



