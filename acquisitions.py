# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from support.processes import Writer

from finance.variables import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


VALUES = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
SCOPE = {"strategy": str, "valuation": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
INDEX = {option: str for option in list(map(str, Securities.Options))}
COLUMNS = {"scenario": str}


class AcquisitionWriter(Writer):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = list(INDEX.keys())
        self.columns = list(COLUMNS.keys())
        self.scope = list(SCOPE.keys())
        self.values = list(VALUES.keys())
        self.valuation = valuation

    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        valuation = str(self.valuation.name).lower()
        header = dict(index=self.index, columns=self.columns, scope=self.scope, values=self.values)
        assert isinstance(valuations, pd.DataFrame)
        mask = valuations["valuation"] == valuations
        valuations = valuation.where(mask).dropna(axis=0, how="all")
        valuations = self.pivot(valuations, *args, **header, delimiter=None, **kwargs)



