# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging

from finance.holdings import HoldingReader, HoldingWriter, HoldingTable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureReader", "DivestitureWriter", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DivestitureTable(HoldingTable, variable="divestitures"): pass
class DivestitureReader(HoldingReader, variable="divestitures"): pass
class DivestitureWriter(HoldingWriter, variable="divestitures"):
    def execute(self, *args, **kwargs):
        pass



