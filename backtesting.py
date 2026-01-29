# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 2025
@name:   Backtesting Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["BackTestingCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class BackTestingCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, technicals, /, **kwargs):
        assert isinstance(technicals, pd.DataFrame)

        with pd.option_context('display.max_rows', 100, 'display.max_columns', None, 'display.min_rows', 75, 'display.float_format', '{:.2f}'.format):
            print(technicals)
            raise Exception()

        return
        yield




