# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd

from finance.holdings import HoldingReader, HoldingWriter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureReader", "DivestitureWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DivestitureReader(HoldingReader): pass
class DivestitureWriter(HoldingWriter):
    def execute(self, contents, *args, **kwargs):
        valuation = str(self.calculation.name).lower()
        valuations, exposures = contents[valuation], contents["exposures"]
        assert isinstance(valuations, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return

        ### Check each valuation strategy against exposures ###


        valuations = valuations.reset_index(drop=True, inplace=False)
        self.write(valuations, *args, **kwargs)



