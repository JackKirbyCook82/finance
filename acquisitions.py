# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd

from finance.holdings import HoldingReader, HoldingWriter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionReader", "AcquisitionWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class AcquisitionReader(HoldingReader): pass
class AcquisitionWriter(HoldingWriter):
    def execute(self, contents, *args, **kwargs):
        valuation = str(self.calculation.name).lower()
        valuations = contents[valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        valuations = valuations.reset_index(drop=True, inplace=False)
        self.write(valuations, *args, **kwargs)



