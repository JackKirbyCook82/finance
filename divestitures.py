# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging

from support.files import DataframeFile
from support.tables import DataframeTable
from support.pipelines import CycleRoutine, Consumer

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureWriter", "DivestitureSaver", "DivestitureFile", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class DivestitureWriter(Consumer):
    def execute(self, query, *args, **kwargs):
        pass


class DivestitureSaver(CycleRoutine):
    def execute(self, *args, **kwargs):
        pass


class DivestitureFile(DataframeFile):
    def dataheader(self, *args, **kwargs): pass
    def datatypes(self, *args, **kwargs): pass
    def datetypes(self, *args, **kwargs): pass


class DivestitureTable(DataframeTable):
    def execute(self, *args, **kwargs):
        pass

    @staticmethod
    def parser(index, record):
        pass

    @property
    def header(self): pass




