# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging

from support.tables import Tables

from finance.holdings import HoldingWriter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureWriter", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DivestitureTable(Tables.Dataframe, variable="divestitures"): pass
class DivestitureWriter(HoldingWriter):
    def execute(self, *args, **kwargs):
        pass



