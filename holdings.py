# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables, Contract
from support.meta import ParametersMeta
from support.mixins import Sizing
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class HoldingParameters(metaclass=ParametersMeta):
    filename = lambda contract: "_".join([str(contract.ticker).upper(), str(contract.expire.strftime("%Y%m%d"))])
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
    dates = {"expire": "%Y%m%d"}


class HoldingVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}
    axes.update({Variables.Instruments.STOCK: list(Variables.Securities.Stocks), Variables.Instruments.OPTION: list(Variables.Securities.Options)})
    data = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}

    def __init__(self, *args, valuation, **kwargs):
        self.options = list(map(str, self.axes[Variables.Instruments.OPTIONS]))
        self.stocks = list(map(str, self.axes[Variables.Instruments.STOCK]))
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.stacking = self.data[valuation]
        self.header = self.contract + self.security + ["strike", "quantity"]
        self.securities = self.options + self.stocks


class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, datatype=pd.DataFrame, **dict(HoldingParameters)):
    pass


class HoldingCalculator(Sizing):
    def __repr__(self): return str(self.name)
    def __init__(self, *args, valuation, **kwargs):
        self.__name = kwargs.pop("name", self.__class__.__name__)
        self.__variables = HoldingVariables(*args, valuation=valuation, **kwargs)
        self.__valuation = valuation
        self.__logger = __logger__

    def __call__(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        for contract, dataframe in self.contracts(valuations):
            holdings = self.execute(dataframe, *args, **kwargs)
            size = self.size(holdings)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if bool(holdings.empty): continue
            yield holdings

    def contracts(self, valuations):
        assert isinstance(valuations, pd.DataFrame)
        for (ticker, expire), dataframe in valuations.groupby(self.variables.contract):
            if bool(dataframe.empty): continue
            contract = Contract(ticker, expire)
            yield contract, dataframe

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.unpivot(valuations, *args, **kwargs)
        holdings = self.calculate(valuations, *args, **kwargs)
        return holdings

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        stocks = self.stocks(valuations, *args, **kwargs)
        securities = pd.concat([valuations, stocks], axis=1)
        holdings = self.holdings(securities, *args, **kwargs)
        return holdings

    def stocks(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in list(self.variables.stocks)}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        return stocks

    def holdings(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        securities = securities[[column for column in list(self.variables.header) if column in securities.columns] + list(self.variables.securities)]
        contracts = [column for column in securities.columns if column not in list(self.variables.securities)]
        holdings = securities.melt(id_vars=contracts, value_vars=list(self.variables.securities), var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[self.variables.header]

    def unpivot(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({"scenario"} | set(self.variables.stacking))
        valuations = dataframe[list(index)].droplevel("scenario", axis=1)
        return valuations

    @property
    def variables(self): return self.__variables
    @property
    def valuation(self): return self.__valuation
    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


