# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd

from finance.targets import TargetReader, TargetWriter, TargetTable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionWriter", "AcquisitionReader", "AcquisitionTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class AcquisitionTable(TargetTable): pass
class AcquisitionReader(TargetReader): pass
class AcquisitionWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, pd.DataFrame)
        market = self.market(valuations, *args, **kwargs)
        market = self.prioritize(market, *args, **kwargs)
        if bool(market.empty):
            return
        self.write(market, *args, **kwargs)








