# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Separating
from support.meta import MappingMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator", "HoldingFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HoldingParameters(metaclass=MappingMeta):
    formatters = {"instrument": int, "option": int, "position": int, "strike": lambda strike: round(strike, 2)}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    order = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity"]
    types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
    dates = {"expire": "%Y%m%d"}


class HoldingFile(File, variable="holdings", **dict(HoldingParameters)):
    pass


class HoldingCalculator(Separating, Sizing, Emptying, Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__query = Querys.Contract

    def execute(self, prospects, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame)
        if self.empty(prospects): return
        for parameters, dataframe in self.separate(prospects, *args, fields=self.fields, **kwargs):
            contract = self.query(parameters)
            holdings = self.calculate(dataframe, *args, **kwargs)
            size = self.size(holdings)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
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
        header = list(Querys.Product) + list(Variables.Security) + ["quantity"]
        columns = [column for column in list(header) if column in dataframe.columns]
#        securities = dataframe[columns + list(map(str, Variables.Securities.Stocks)) + list(map(str, Variables.Securities.Options))]
#        holdings = securities.melt(id_vars=list(Querys.Contract), value_vars=list(map(str, Variables.Securities)), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
#        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[header]

    @staticmethod
    def stocks(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
#        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in list(map(str, Variables.Securities.Stocks))}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        return stocks

    @property
    def fields(self): return list(self.__query)
    @property
    def query(self): return self.__query

