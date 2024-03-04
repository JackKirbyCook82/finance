# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
from abc import ABC
from collections import OrderedDict as ODict

from support.files import DataframeFile
from support.processes import Calculator
from support.calculations import Calculation
from support.processes import Saver, Loader, Filter, Parser

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {}
COLUMNS = {}


class PortfolioCalculation(Calculation, ABC, fields=[]):
    pass


class PortfolioCalculator(Calculator, calculations=ODict(list(PortfolioCalculation))):
    def execute(self, query, *args, **kwargs):
        pass


class PortfolioFilter(Filter, index=INDEX, columns=COLUMNS):
    def execute(self, query, *args, **kwargs):
        pass


class PortfolioParser(Parser, index=INDEX, columns=COLUMNS):
    def execute(self, query, *args, **kwargs):
        pass


class PortfolioFile(DataframeFile, index=INDEX, columns=COLUMNS): pass
class PortfolioSaver(Saver):
    def execute(self, query, *args, **kwargs):
        pass


class PortfolioLoader(Loader):
    def execute(self, *args, **kwargs):
        pass



