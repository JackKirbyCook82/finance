# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 15 2024
@name:   Divestitures Objects
@author: Jack Kirby Cook

"""

from finance.holdings import HoldingReader, HoldingWriter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class DivestitureWriter(HoldingWriter):
    def execute(self, contents, *args, **kwargs):
        pass


class DivestitureReader(HoldingReader):
    def execute(self, *args, **kwargs):
        pass





