# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 2025
@name:   Risk Objects
@author: Jack Kirby Cook

"""

import xarray as xr
from abc import ABC

from support.mixins import Emptying, Sizing, Partition, Logging
from support.calculations import Calculation, Equation, Variable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["RiskCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class RiskEquation(Equation, ABC, datatype=xr.DataArray, vectorize=True):
    def execute(self, *args, **kwargs):
        yield from super().execute(*args, **kwargs)


class RiskCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, strategies, *args, **kwargs):
        print(strategies)
        raise Exception()



