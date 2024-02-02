# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
from datetime import datetime as Datetime

from support.files import DataframeFile
from support.tables import DataframeTable
from support.pipelines import CycleRoutine, Consumer

from finance.variables import Securities, Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionWriter", "AcquisitionSaver", "AcquisitionFile", "AcquisitionTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class AcquisitionWriter(Consumer):
    def __init__(self, *args, table, **kwargs):
        assert isinstance(table, AcquisitionTable)
        super().__init__(*args, **kwargs)
        self.table = table

    def execute(self, query, *args, liquidity=None, apy=None, **kwargs):
        if not bool(query.arbitrages) or all([dataframe.empty for dataframe in query.arbitrages.values()]):
            return
        targets = query.arbitrages[Valuations.Arbitrage.Minimum]
        liquidity = liquidity if liquidity is not None else 1
        targets["size"] = (targets["size"] * liquidity).apply(np.floor).astype(np.int32)
        targets = targets.where(targets["size"] > 0)
        targets = targets.where(targets["apy"] >= apy) if apy is not None else targets
        targets = targets.dropna(axis=0, how="all")
        targets = targets.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        if bool(targets.empty):
            return
        assert targets["apy"].min() > 0 and targets["size"].min() > 0
        targets = targets.reset_index(drop=True, inplace=False)
        targets["size"] = targets["size"].astype(np.int32)
        targets["tau"] = targets["tau"].astype(np.int32)
        targets["apy"] = targets["apy"].round(2)
        targets["npv"] = targets["npv"].round(2)
        targets["inquiry"] = query.inquiry


class AcquisitionSaver(CycleRoutine):
    def __init__(self, *args, table, file, **kwargs):
        assert isinstance(table, AcquisitionTable)
        super().__init__(*args, **kwargs)
        self.table = table
        self.file = file

    def execute(self, *args, **kwargs):
        pass


class AcquisitionFile(DataframeFile):
    def dataheader(self, *args, **kwargs): pass
    def datatypes(self, *args, **kwargs): pass
    def datetypes(self, *args, **kwargs): pass


class AcquisitionTable(DataframeTable):
    def execute(self, *args, funds=None, tenure=None, **kwargs):
        self.table = self.table.where(self.table["inquiry"] - Datetime.now() < tenure) if tenure is not None else self.table
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
    def parser(index, record):
        pass

    @property
    def header(self): return ["inquiry", "strategy", "ticker", "expire"] + list(map(str, Securities.Options)) + ["apy", "tau", "npv", "cost", "size"]



