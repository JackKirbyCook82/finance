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

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class PricingEquation(Equation, ABC):
    j = Variable("j", "position", Variables.Securities.Position, types.NoneType, locator="position")
    x = Variable("x", "underlying", np.float32, pd.Series, locator="underlying")
    t = Variable("t", "current", np.datetime64, pd.Series, locator="current")

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
            yield equation.x()
            yield equation.q()
            yield equation.t()

class MarketCalculation(PricingCalculation, equation=MarketEquation, register=Variables.Markets.Pricing.MARKET): pass
class LimitCalculation(PricingCalculation, equation=LimitEquation, register=Variables.Markets.Pricing.LIMIT): pass
class CenteredCalculation(PricingCalculation, equation=CenteredEquation, register=Variables.Markets.Pricing.CENTERED): pass


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = dict(PricingCalculation)[pricing](*args, **kwargs)
        self.__pricing = pricing

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        for settlement, dataframe in self.partition(options, by=Querys.Settlement):
            securities = self.calculate(dataframe, *args, **kwargs)
            size = self.size(securities)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(securities): continue
            yield securities

    def calculate(self, options, *args, **kwargs):
        securities = dict(self.calculator(options, *args, **kwargs))
        securities = pd.concat(list(securities.values()), axis=0)
        return securities

    def calculator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for position in list(Variables.Securities.Position):
            securities = self.calculation(options, *args, position=position, **kwargs)
            securities = pd.concat([options[list(Querys.Contract)], securities], axis=1)
            securities["instrument"] = Variables.Securities.Instrument.OPTION
            securities["position"] = position
            securities = securities.reset_index(drop=True, inplace=False)
            yield position, securities

    @property
    def calculation(self): return self.__calculation
    @property
    def pricing(self): return self.__pricing



