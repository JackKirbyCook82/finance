# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from functools import reduce
from abc import ABC, abstractmethod

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StockCalculator", "OptionCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


enumerical = lambda integer: Variables.Securities.Position.LONG if integer > 0 else Variables.Securities.Position.SHORT
numerical = lambda position: 2 * int(bool(position is Variables.Securities.Position.LONG)) - 1


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


class VirtualCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    pass


class SecurityCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = dict(PricingCalculation)[pricing](*args, **kwargs)
        self.__pricing = pricing

    @staticmethod
    @abstractmethod
    def quantity(series, *args, securities, **kwargs): pass
    @abstractmethod
    def execute(self, securities, *args, **kwargs): pass
    @abstractmethod
    def calculate(self, securities, *args, **kwargs): pass
    @abstractmethod
    def calculator(self, securities, *args, **kwargs): pass

    @property
    def calculation(self): return self.__calculation
    @property
    def pricing(self): return self.__pricing


class StockCalculator(SecurityCalculator, title="Calculated"):
    def execute(self, stocks, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
        if self.empty(stocks): return
        for symbol, primary in self.partition(stocks, by=Querys.Symbol):
            results = self.calculate(primary, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(symbol)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, stocks, *args, **kwargs):
        securities = dict(self.calculator(stocks, *args, **kwargs))
        securities = pd.concat(list(securities.values()), axis=0)
        securities["quantity"] = securities.apply(self.quantity, axis=1, stocks=stocks)
        securities = securities.reset_index(drop=True, inplace=False)
        return securities

    def calculator(self, stocks, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame)
        for position in list(Variables.Securities.Position):
            pricing = self.calculation(stocks, *args, position=position, **kwargs)
            securities = pd.concat([stocks[list(Querys.Symbol)], pricing], axis=1)
            securities["instrument"] = Variables.Securities.Instrument.STOCK
            securities["position"] = position
            yield position, securities

    @staticmethod
    def quantity(series, *args, stocks, **kwargs):
        if "quantity" not in stocks.columns: return 0
        contents = series[list(Querys.Symbol)].to_dict().items()
        mask = [stocks[key] == value for key, value in contents]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        stock = stocks.where(mask).dropna(how="all", inplace=False).squeeze()
        enumerically = enumerical(np.sign(stock.quantity)) == series.position
        numerically = np.sign(stock.quantity) == numerical(series.position)
        assert enumerically == numerically
        return stock.quantity if (numerically & enumerically) else 0


class OptionCalculator(SecurityCalculator, title="Calculated"):
    def execute(self, stocks, options, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame)
        if self.empty(stocks) and self.empty(options): return
        for settlement, primary in self.partition(options, by=Querys.Settlement):
            secondary = stocks.where(stocks["ticker"] == settlement.ticker).dropna(how="all", inplace=False)
            results = self.calculate(primary, *args, stocks=secondary, **kwargs)
            size = self.size(results)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, options, *args, stocks, **kwargs):
        securities = dict(self.calculator(options, *args, **kwargs))
        securities = pd.concat(list(securities.values()), axis=0)
        securities["quantity"] = securities.apply(self.quantity, axis=1, options=options)
        securities["underlying"] = np.round(stocks.squeeze().price, 2)
        securities = securities.reset_index(drop=True, inplace=False)
        return securities

    def calculator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for position in list(Variables.Securities.Position):
            pricing = self.calculation(options, *args, position=position, **kwargs)
            securities = pd.concat([options[list(Querys.Contract)], pricing], axis=1)
            securities["instrument"] = Variables.Securities.Instrument.OPTION
            securities["position"] = position
            yield position, securities

    @staticmethod
    def quantity(series, *args, options, **kwargs):
        if "quantity" not in options.columns: return 0
        contents = series[list(Querys.Contract)].to_dict().items()
        mask = [options[key] == value for key, value in contents]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        option = options.where(mask).dropna(how="all", inplace=False).squeeze()
        enumerically = enumerical(np.sign(option.quantity)) == series.position
        numerically = np.sign(option.quantity) == numerical(series.position)
        assert enumerically == numerically
        return option.quantity if (numerically & enumerically) else 0

