# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 2 2025
@name:   Orders Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from finance.variables import Querys
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OrderCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Order():
    pass


class OrderCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        if self.empty(prospects): return

        print(prospects)
        raise Exception()

        for settlement, dataframe in self.partition(prospects, by=Querys.Settlement):
            pass