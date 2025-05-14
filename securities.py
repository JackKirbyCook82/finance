# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Securities Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from finance.variables import Querys
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, stocks, options, technicals, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame)
        if self.empty(options): return

        print(stocks)
        print(technicals)
        print(options)
        raise Exception()



