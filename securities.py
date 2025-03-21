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


class SecurityCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__instrument__ = kwargs.get("instrument", getattr(cls, "__instrument__", None))
        cls.__header__ = kwargs.get("header", getattr(cls, "__header__", None))
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = dict(PricingCalculation)[pricing](*args, **kwargs)
        self.__pricing = pricing

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        for query, dataframe in self.partition(securities, by=self.query):
            results = self.calculate(dataframe, *args, **kwargs)
            size = self.size(results)
            self.console(f"{str(query)}[{int(size):.0f}]")
            if self.empty(results): continue
            yield results

    def calculate(self, securities, *args, **kwargs):
        mapping = dict(self.calculator(securities, *args, **kwargs))
        dataframe = pd.concat(list(mapping.values()), axis=0)
        if "quantity" in securities.columns:
            dataframe["quantity"] = dataframe.apply(self.quantify, axis=1, securities=securities, **kwargs)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    def calculator(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        for position in list(Variables.Securities.Position):
            pricing = self.calculation(securities, *args, position=position, **kwargs)
            dataframe = pd.concat([securities[self.header], pricing], axis=1)
            dataframe["instrument"] = self.instrument
            dataframe["position"] = position
            yield position, dataframe

    def quantify(self, series, *args, securities, **kwargs):
        assert "quantity" in securities.columns
        contents = series[self.header].to_dict().items()
        mask = [securities[key] == value for key, value in contents]
        mask = reduce(lambda lead, lag: lead & lag, mask)
        security = securities.where(mask).dropna(how="all", inplace=False).squeeze()
        enumerically = enumerical(np.sign(security.quantity)) == series.position
        numerically = np.sign(security.quantity) == numerical(series.position)
        assert enumerically == numerically
        return np.abs(security.quantity) if (numerically & enumerically) else 0

    @property
    def instrument(self): return type(self).__instrument__
    @property
    def header(self): return type(self).__header__
    @property
    def query(self): return type(self).__query__
    @property
    def calculation(self): return self.__calculation
    @property
    def pricing(self): return self.__pricing


class StockCalculator(SecurityCalculator, query=Querys.Symbol, instrument=Variables.Securities.Instrument.STOCK, header=list(Querys.Symbol)): pass
class OptionCalculator(SecurityCalculator, query=Querys.Settlement, instrument=Variables.Securities.Instrument.OPTION, header=list(Querys.Contract)): pass

