# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging

from support.tables import DataframeTable
from support.pipelines import Consumer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureWriter", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DivestitureWriter(Consumer):
    def execute(self, query, *args, **kwargs):
        pass


class DivestitureTable(DataframeTable):
    def execute(self, *args, **kwargs):
        pass

    @staticmethod
    def parser(index, record):
        pass

    @property
    def header(self): pass




