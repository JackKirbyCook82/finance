# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Market Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC

from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator", "DivestitureCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class MarketCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init__(self, *args, liquidity, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(liquidity) and callable(priority)
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, stocks, options, valuations, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame) and isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return

        print("\n", valuations, "\n")
        print(options, "\n")
        print(stocks, "\n")
        raise Exception()

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority


class AcquisitionCalculator(MarketCalculator):
    pass


class DivestitureCalculator(MarketCalculator):
    pass


