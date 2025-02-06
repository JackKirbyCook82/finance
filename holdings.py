# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import MappingMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator", "HoldingFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class HoldingParameters(metaclass=MappingMeta):
    order = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity"]
    types = {"ticker": str, "strike": np.float32, "quantity": np.float32}
    parsers = dict(instrument=Variables.Securities.Instrument, option=Variables.Securities.Option, position=Variables.Securities.Position)
    formatters = dict(instrument=int, option=int, position=int)
    dates = dict(expire="Y%m%d")

class HoldingFile(File, **dict(HoldingParameters)):
    pass


class HoldingCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        if self.empty(prospects): return
        for settlement, dataframe in self.partition(prospects, by=Querys.Settlement):
            holdings = self.calculate(dataframe, *args, **kwargs)
            size = self.size(holdings)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(holdings): continue
            yield holdings

    def calculate(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        stocks = self.stocks(prospects, *args, **kwargs)
        dataframe = pd.concat([prospects, stocks], axis=1)
        holdings = self.holdings(dataframe, *args, **kwargs)
        return holdings

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        header = list(Querys.Settlement) + ["strike"] + list(Variables.Securities.Security) + ["quantity"]
        columns = [column for column in list(header) if column in dataframe.columns]
        securities = dataframe[columns + list(map(str, Securities.Stocks)) + list(map(str, Securities.Options))]
        holdings = securities.melt(id_vars=list(Querys.Settlement), value_vars=list(map(str, Securities)), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[header]

    @staticmethod
    def stocks(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in list(map(str, Securities.Stocks))}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        return stocks



