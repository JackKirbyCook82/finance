# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 2025
@name:   Backtesting Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from finance.concepts import Querys
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
        if self.empty(technicals): return
        symbols = self.keys(technicals, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        backtesting = self.calculate(technicals, **kwargs)
        size = self.size(backtesting)
        self.console(f"{str(symbols)}[{int(size):.0f}]")
        if self.empty(backtesting): return
        yield backtesting

    def calculate(self, technicals, /, **kwargs):
        assert isinstance(technicals, pd.DataFrame)
        technicals = list(self.values(technicals, by=Querys.Symbol))
        backtesting = list(self.calculator(technicals, **kwargs))
        backtesting = pd.concat(backtesting, axis=0)
        backtesting = backtesting.reset_index(drop=True, inplace=False)
        return backtesting

    def calculator(self, technicals, /, **kwargs):
        assert isinstance(technicals, list) and all([isinstance(dataframe, pd.DataFrame) for dataframe in technicals])
        for dataframe in technicals:
            assert (dataframe["ticker"].to_numpy()[0] == dataframe["ticker"]).all()
            dataframe = dataframe.sort_values("date", ascending=True, inplace=False)

            with pd.option_context('display.max_rows', 100, 'display.max_columns', None, 'display.min_rows', 75, 'display.float_format', '{:.2f}'.format):
                print(dataframe)
                raise Exception()

