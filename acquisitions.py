# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd

from support.tables import Tables

from finance.holdings import HoldingWriter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionWriter", "AcquisitionTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class AcquisitionTable(Tables.Dataframe, variable="acquisitions"): pass
class AcquisitionWriter(HoldingWriter):
    def execute(self, query, *args, **kwargs):
        valuations = query["valuations"]
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        self.write(valuations, *args, **kwargs)



