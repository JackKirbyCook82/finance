# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr

from support.pipelines import Calculator
from support.calculations import Calculation, feed, equation, calculation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class ValuationCalculation(Calculation):
    pass


calculations = {}
class ValuationCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        for calculation in iter(self.calculations):
            valuations = calculation(dataset, *args, **kwargs)
            yield ticker, expire, strategy, valuations


