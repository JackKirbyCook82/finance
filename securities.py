# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC
from functools import reduce

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta, ParameterMeta
from support.variables import Category
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityCalculator", "SecurityFiles"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


enumerical = lambda integer: Variables.Securities.Position.LONG if integer > 0 else Variables.Securities.Position.SHORT
numerical = lambda position: 2 * int(bool(position is Variables.Securities.Position.LONG)) - 1


class SecurityParameters(metaclass=ParameterMeta):
    types = {"ticker": str, "price bid ask": np.float32, "size supply demand": np.float32, "strike underlying": np.float32}
    parsers = dict(instrument=Variables.Securities.Instrument, option=Variables.Securities.Option, position=Variables.Securities.Position)
    formatters = dict(instrument=int, option=int, position=int)
    dates = dict(date="%Y%m%d", expire="%Y%m%d", current="%Y%m%d-%H%M")

class SecurityFile(File, **dict(SecurityParameters)): pass
class StockTradeFile(SecurityFile, order=["ticker", "price"]): pass
class StockQuoteFile(SecurityFile, order=["ticker", "bid", "ask", "demand", "supply"]): pass
class StockSecurityFile(SecurityFile, order=["ticker", "position", "price", "size"]): pass
class OptionTradeFile(SecurityFile, order=["ticker", "expire", "strike", "option", "price"]): pass
class OptionQuoteFile(SecurityFile, order=["ticker", "expire", "strike", "option", "bid", "ask", "demand", "supply"]): pass
class OptionSecurityFile(SecurityFile, order=["ticker", "expire", "strike", "instrument", "option", "position", "price", "underlying", "size"]): pass

class SecurityFiles(Category):
    class Stocks(Category): Trade, Quote, Security = StockTradeFile, StockQuoteFile, StockSecurityFile
    class Options(Category): Trade, Quote, Security = OptionTradeFile, OptionQuoteFile, OptionSecurityFile


class PricingEquation(Equation, ABC):
    j = Variable("j", "position", Variables.Securities.Position, types.NoneType, locator="position")

    qα = Variable("qα", "supply", np.float32, pd.Series, locator="supply")
    qβ = Variable("qβ", "demand", np.float32, pd.Series, locator="demand")
    yα = Variable("yα", "ask", np.float32, pd.Series, locator="ask")
    yβ = Variable("yβ", "bid", np.float32, pd.Series, locator="bid")

class MarketEquation(PricingEquation):
    y = Variable("y", "price", np.float32, pd.Series, vectorize=False, function=lambda yα, yβ, j: {Variables.Securities.Position.LONG: yα, Variables.Securities.Position.SHORT: yβ}[j])
    q = Variable("q", "size", np.float32, pd.Series, vectorize=False, function=lambda qα, qβ, j: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[j])

class LimitEquation(PricingEquation):
    y = Variable("y", "price", np.float32, pd.Series, vectorize=False, function=lambda yα, yβ, j: {Variables.Securities.Position.LONG: yβ, Variables.Securities.Position.SHORT: yα}[j])
    q = Variable("q", "size", np.float32, pd.Series, vectorize=False, function=lambda qα, qβ, j: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[j])

class CenteredEquation(PricingEquation):
    y = Variable("y", "price", np.float32, pd.Series, vectorize=False, function=lambda yα, yβ: (yα + yβ) / 2)
    q = Variable("q", "size", np.float32, pd.Series, vectorize=False, function=lambda qα, qβ, j: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[j])


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta):
    def execute(self, securities, *args, position, **kwargs):
        with self.equation(securities, position=position) as equation:
            yield equation.y()
            yield equation.q()

class AggressiveCalculation(PricingCalculation, equation=MarketEquation, register=Variables.Markets.Pricing.AGGRESSIVE): pass
class PassiveCalculation(PricingCalculation, equation=LimitEquation, register=Variables.Markets.Pricing.PASSIVE): pass
class ModerateCalculation(PricingCalculation, equation=CenteredEquation, register=Variables.Markets.Pricing.MODERATE): pass


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = dict(PricingCalculation)[pricing](*args, **kwargs)
        self.__pricing = pricing

    def execute(self, stocks, options, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame)
        if self.empty(stocks) and self.empty(options): return
        for settlement, derivatives in self.partition(options, by=Querys.Settlement):
            assets = stocks.where(stocks["ticker"] == settlement.ticker).dropna(how="all", inplace=False)
            securities = self.calculate(derivatives, *args, stocks=assets, **kwargs)
            size = self.size(securities)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(securities): continue
            yield securities

    def calculate(self, options, *args, stocks, **kwargs):
        securities = dict(self.calculator(options, *args, **kwargs))
        securities = pd.concat(list(securities.values()), axis=0)
        securities["owned"] = securities.apply(self.derivatives, axis=1, options=options)
        securities["shares"] = securities.apply(self.assets, axis=1, stocks=stocks)
        securities["underlying"] = np.round(stocks.squeeze().price, 2)
        securities = securities.reset_index(drop=True, inplace=False)
        return securities

    def calculator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for position in list(Variables.Securities.Position):
            securities = self.calculation(options, *args, position=position, **kwargs)
            securities = pd.concat([options[list(Querys.Contract)], securities], axis=1)
            securities["instrument"] = Variables.Securities.Instrument.OPTION
            securities["position"] = position
            yield position, securities

    @staticmethod
    def derivatives(series, *args, options, **kwargs):
        if "quantity" not in options.columns: return 0
        contents = series[list(Querys.Contract)].to_dict().items()
        mask = [options[key] == value for key, value in contents]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        option = options.where(mask).dropna(how="all", inplace=False).squeeze()
        enumerically = enumerical(np.sign(option.quantity)) == series.position
        numerically = np.sign(option.quantity) == numerical(series.position)
        assert enumerically == numerically
        return option.quantity if (numerically & enumerically) else 0

    @staticmethod
    def assets(series, *args, stocks, **kwargs):
        if "quantity" not in stocks.columns: return 0
        contents = series[list(Querys.Symbol)].to_dict().items()
        mask = [stocks[key] == value for key, value in contents]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        stock = stocks.where(mask).dropna(how="all", inplace=False).squeeze()
        return stock.quantity

    @property
    def calculation(self): return self.__calculation
    @property
    def pricing(self): return self.__pricing



