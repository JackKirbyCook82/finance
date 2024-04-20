# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging

from finance.holdings import HoldingReader, HoldingWriter, HoldingTable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionReader", "AcquisitionWriter", "AcquisitionTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class AcquisitionTable(HoldingTable, variable="acquisitions"): pass
class AcquisitionReader(HoldingReader, variable="acquisitions"): pass
class AcquisitionWriter(HoldingWriter, variable="acquisitions"):
    def execute(self, query, *args, **kwargs):
        valuations = query["valuations"]
        if self.empty(valuations):
            return
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        self.write(valuations, *args, **kwargs)



