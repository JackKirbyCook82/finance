# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging

from finance.targets import TargetReader, TargetWriter, TargetTable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureWriter", "DivestitureReader", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DivestitureTable(TargetTable): pass
class DivestitureReader(TargetReader): pass
class DivestitureWriter(TargetWriter):
    def execute(self, *args, **kwargs):
        pass



