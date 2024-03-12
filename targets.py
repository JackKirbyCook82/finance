# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC
from collections import namedtuple as ntuple

from support.processes import Writer

from finance.variables import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionWriter", "DivestitureWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


VALUES = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
SCOPE = {"strategy": str, "valuation": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
INDEX = {option: str for option in list(map(str, Securities.Options))}
COLUMNS = {"scenario": str}

Contract = ntuple("Contract", "ticker expire")
Option = ntuple("Content", "instrument position strike")
Holding = ntuple("Holding", "contract option")


class TargetWriter(Writer, ABC):
    def __init__(self, *args, valuation, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(priority)
        self.index = list(INDEX.keys())
        self.columns = list(COLUMNS.keys())
        self.scope = list(SCOPE.keys())
        self.values = list(VALUES.keys())
        self.valuation = valuation

    def parse(self, dataframe, *args, liquidity=None, apy=None, **kwargs):
        header = dict(index=self.index, columns=self.columns, scope=self.scope, values=self.values)
        liquidity = liquidity if liquidity is not None else 1
        apy = apy if apy is not None else 0
        function = lambda cols: (np.min(cols.values) * liquidity).apply(np.floor).astype(np.int32)
        mask = dataframe["valuation"] == str(self.valuation.name).lower()
        dataframe = dataframe.where(mask).dropna(axis=0, how="all")
        scenarios = set(dataframe["scenario"].values)
        dataframe = self.pivot(dataframe, *args, **header, **kwargs)
        columns = [(scenario, "size") for scenario in scenarios]
        dataframe["liquidity"] = dataframe[columns].apply(function)
        mask = dataframe["liquidity"] > liquidity & dataframe["apy"] > apy
        dataframe = dataframe.where(mask).dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    def prioritize(self, dataframe, *args, **kwargs):
        dataframe["priority"] = dataframe.apply(self.priority)
        dataframe = dataframe.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        return dataframe

    def write(self, dataframe, *args, **kwargs):
        pass


class AcquisitionWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.parse(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        valuations = self.prioritize(valuations, *args, **kwargs)
        self.write(valuations, *args, **kwargs)


class DivestitureWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations, holdings = query.valuations, query.holdings
        assert isinstance(valuations, pd.DataFrame) and isinstance(holdings, pd.DataFrame)
        valuations = self.parse(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        valuations = self.prioritize(valuations, *args, **kwargs)
        holdings = self.options(holdings, *args, **kwargs)



    def options(self, dataframe, *args, **kwargs):
        pass








