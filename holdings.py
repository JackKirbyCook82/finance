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
from support.mixins import Sizing, Logging, Pipelining, Sourcing
from support.meta import ParametersMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HoldingParameters(metaclass=ParametersMeta):
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
    dates = {"expire": "%Y%m%d"}
#    queryname = lambda filename: Contract.fromstring(filename)
#    filename = lambda queryname: Contract.tostring(queryname)


# class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, datatype=pd.DataFrame, **dict(HoldingParameters)):
#     pass


class HoldingCalculator(Pipelining, Sourcing, Sizing, Logging):
    def __init__(self, *args, valuation, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        index = list(Querys.Product.fields) + list(Variables.Security.fields)
        valuations = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
        self.__header = list(index) + ["quantity"]
        self.__stacking = valuations[valuation]

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        for contract, dataframe in self.source(valuations, Querys.Contract):
            if self.empty(dataframe): continue
            holdings = self.calculate(dataframe, *args, **kwargs)
            size = self.size(holdings)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(holdings): continue
            yield holdings

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.unpivot(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        securities = pd.concat([valuations, stocks], axis=1)
        holdings = self.holdings(securities, *args, **kwargs)
        return holdings

    @staticmethod
    def stocks(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        stocks = list(Variables.Securities.Stocks)
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        return stocks

    def holdings(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        securities = securities[[column for column in list(self.header) if column in securities.columns] + list(Variables.Securities)]
        contracts = [column for column in securities.columns if column not in list(Variables.Securities)]
        holdings = securities.melt(id_vars=contracts, value_vars=list(Variables.Securities), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[self.variables.header]

    def unpivot(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({"scenario"} | set(self.stacking))
        valuations = dataframe[list(index)].droplevel("scenario", axis=1)
        return valuations

    @property
    def stacking(self): return self.__stacking
    @property
    def header(self): return self.__header



